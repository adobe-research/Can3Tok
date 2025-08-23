import torch
import torch.nn.functional as F
import torch.nn as nn
from gaussian_renderer import render, network_gui
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
import os
from plyfile import PlyData, PlyElement
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from utils.loss_utils import l1_loss
from random import randint
import random
from chamferdist import ChamferDistance
from sinkhorn import sinkhorn
from geomloss import SamplesLoss
from voxelize import voxelize
from Michelangelo.michelangelo import *
# import Michelangelo.michelangelo.models.tsal.sal_perceiver.AlignedShapeLatentPerceiver as Perceiver_encoder
from Michelangelo.michelangelo.models.tsal.tsal_base import ShapeAsLatentModule
from Michelangelo.michelangelo.utils import instantiate_from_config
from Michelangelo.michelangelo.utils.misc import get_config_from_file
from gs_dataset_original import gs_dataset
import argparse
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.utils.data as Data
from scipy.stats import special_ortho_group
from sklearn.manifold import TSNE
from scipy.spatial.transform import Rotation as RRR
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl


######### multi GPU setting
# parser = argparse.ArgumentParser()
# parser.add_argument("--local-rank", type=int, default=0)
# args = parser.parse_args()
# local_rank = args.local_rank
# local_rank=0
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
# torch.cuda.set_device(local_rank)
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4,5,6,7"  
# DEVICE = torch.device("cuda", local_rank)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_usage = "L1" # L1, sinkhorn, chamfer
random_permute = 0
random_rotation = 1
random_shuffle = 1

resol = 128 #128
data_path = f"/mnt/localssd/dl3dv-1k/"
# data_path = f"/mnt/localssd/dl3dv-316/"
dummy_image_path = "/home/qgao/sensei-fs-link/Dataset/scripts/DL3DV-10K-Benchmark/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/gaussian_splat/"

folder_path_each = os.listdir(data_path)
# folder_path_each.remove('.ipynb_checkpoints')
num_epochs = 200000
save_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/train_pointvae_{resol}/"  # train_vae    train_ae_only2
# save_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/filter_train_pointvae/"
bch_size = 200 #100
k_rendering_loss = 1000
enable_rendering_loss = 0
label_gt = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).to(device)
L2=torch.nn.CrossEntropyLoss()
LBCE = torch.nn.BCELoss()
chamferDist = ChamferDistance()
sinkhorn_eff = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
class GroupParams:
    pass

def group_extract(param_list, param_value):
    group = GroupParams()
    for idx in range(len(param_list)):
        setattr(group, param_list[idx], param_value[idx])
    return group

model_params_list = ["sh_degree", "source_path", "model_path", "images", "resolution", "white_background", "data_device", "num_gs_per_scene_end", "eval"]
model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", 256, False]
pipeline_params_list = ["convert_SHs_python", "compute_cov3D_python", "debug"]
pipeline_params_value = [False, False, False]
optimization_params_list = ["iterations", "position_lr_init", "position_lr_final", "position_lr_delay_mult", "position_lr_max_steps",
                               "feature_lr", "opacity_lr", "scaling_lr", "rotation_lr", "percent_dense", "lambda_dssim", 
                               "densification_interval", "opacity_reset_interval", "densify_from_iter", "densify_until_iter", 
                               "densify_grad_threshold", "random_background"]
optimization_params_value = [35_000, 0.00016, 0.0000016, 0.01, 30_000, 0.0025, 0.05, 0.005, 0.001, 0.01, 0.2, 100, 3000, 500, 15_000,
                                0.0002, False]

viewpoint_stack = []
for idx_batch in range(0,1):
    # dummy_image_path = "/home/qgao/sensei-fs-link/Dataset/scripts/DL3DV-10K-Benchmark/" + folder_path_each[idx_batch] + "/gaussian_splat/"
    model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", 256, False]
    dataset_for_gs = group_extract(model_params_list, model_params_value)
    gaussians = GaussianModel(dataset_for_gs.sh_degree)
    scene = Scene(dataset_for_gs, gaussians)
    # train_dataset = scene.getTrainCameras()
    viewpoint_stack.append(scene.getTrainCameras().copy())
    training_setup_for_gs = group_extract(optimization_params_list, optimization_params_value)
    pipe = group_extract(pipeline_params_list, pipeline_params_value)



background = torch.tensor([0,0,0], dtype=torch.float32).to(device)


######## perciever VAE
config_path_perciever = "./Michelangelo/configs/aligned_shape_latents/shapevae-256.yaml"
model_config_perciever = get_config_from_file(config_path_perciever)
if hasattr(model_config_perciever, "model"):
   model_config_perciever = model_config_perciever.model
perceiver_encoder_decoder = instantiate_from_config(model_config_perciever)


################# train
ckpt = 0
if torch.cuda.device_count() > 1:
  gs_autoencoder = nn.DataParallel(perceiver_encoder_decoder)
else:
  gs_autoencoder = perceiver_encoder_decoder
gs_autoencoder.to(device)
# #################


################# toy eval
ckpt = 110000
subpath = f"{ckpt}.pth"
# gs_autoencoder = perceiver_encoder_decoder.to(device)
# gs_autoencoder.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(save_path, os.path.join(save_path, subpath))).items()})
gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, os.path.join(save_path, subpath))),strict=True)

gs_autoencoder.eval()
# gs_autoencoder = nn.DataParallel(gs_autoencoder)
# exit()
#################

optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=1e-5, betas=[0.9, 0.999])

#########################################
#### reconstruction check
gs_dataset = gs_dataset(data_path, resol = 128, random_permute = False, train=True)
trainDataLoader = Data.DataLoader(dataset=gs_dataset, batch_size=bch_size, shuffle=True, num_workers=12) 

gs_autoencoder.eval()

scene_num = 2*bch_size
z_chunk = []
ikik = 0
tsne_gs = torch.zeros(scene_num, 16384, 18).to(device)
for i_batch, UV_gs_batch_raw in enumerate(trainDataLoader):
    UV_gs_batch = UV_gs_batch_raw[0].to(dtype = torch.float32).to(device)
    if ikik == 0:
      UV_gs_batch[:,:,:] = UV_gs_batch[0,:,:]
      for tsne_exp in range(bch_size):
          #### random rotation
          # rand_rot_comp = special_ortho_group.rvs(3)
          # rand_rot = torch.tensor(np.dot(rand_rot_comp, rand_rot_comp.T), dtype = torch.float32).to(UV_gs_batch.device)
          # tsne_gs[tsne_exp,:,4:7] = UV_gs_batch[tsne_exp,:,4:7]@rand_rot
          #### rotation with an angle
          angle_degrees = 10*tsne_exp
          axis = [0,1,0]  # x,y,z-axis
          angle_radians = np.radians(angle_degrees)
          rotation_gen = torch.tensor(RRR.from_rotvec(angle_radians * np.array(axis)).as_matrix(), dtype=torch.float32).to(UV_gs_batch.device)
          tsne_gs[tsne_exp,:,4:7] = UV_gs_batch[tsne_exp,:,4:7]@rotation_gen
          if tsne_exp == 36:
              break
        
      ##
      # tsne_gs[:bch_size] = UV_gs_batch
      ##
      shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(tsne_gs[:bch_size],tsne_gs[:bch_size],tsne_gs[:bch_size],tsne_gs[:bch_size])
      z_chunk.append(z)
    else:  
      shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(UV_gs_batch,UV_gs_batch,UV_gs_batch,UV_gs_batch)
      z_chunk.append(z)
    ikik += 1
    if ikik == 1: # 2
      break

z = torch.concatenate(z_chunk,dim=0)
#### random rotation
# label_Y = [0 for i in range(bch_size)] + [1 for j in range(bch_size)]
# data_X = np.array(z.view(z.shape[0], -1).detach().cpu())
# colors_candidates = ["b", "r", "o"]
# colors = [colors_candidates[0] for i in range(bch_size)] + [colors_candidates[1] for j in range(bch_size)]
#### rotation with an angle
z = z[:36]
fig = plt.figure()
display_axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
label_Y = [i for i in range(bch_size)]
data_X = np.array(z.view(z.shape[0], -1).detach().cpu())
quant_steps = 36
norm = mpl.colors.Normalize(0, 2 * np.pi)
hsv = cm.get_cmap('hsv', quant_steps)
color_bar = hsv(np.tile(np.linspace(0,1,quant_steps),1))
cmap = mpl.colors.ListedColormap(color_bar)
cb = mpl.colorbar.ColorbarBase(display_axes,
                               cmap=cmap,
                               norm=norm,
                               orientation='horizontal')
cb.outline.set_visible(False)                                 
display_axes.set_axis_off()
plt.show()
plt.savefig("color_map.png")
X_embedded = TSNE(n_components=2,init="pca",perplexity=5).fit_transform(data_X)
figure=plt.figure(figsize=(5,5),dpi=80)

x=X_embedded[:,0]
y=X_embedded[:,1]   


### original 200 vs 200
# plt.scatter(x,y,color=colors)
# plt.savefig("tsne_200vs200.png")
### mixed
# plt.scatter(x,y,color=colors)
# plt.savefig("tsne.png")
### color bar vs rotation
plt.scatter(x,y,color=color_bar)
plt.show()
plt.savefig("tsne_rotation.png")
exit()

    
    
print(f"test_error = {torch.norm(UV_gs_recover.reshape(UV_gs_batch.shape[0],-1,14) - UV_gs_batch[:,:,4:], p=2)/UV_gs_batch.shape[0]}")
  

