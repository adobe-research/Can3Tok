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
from model.michelangelo import *
from model.michelangelo.models.tsal.tsal_base import ShapeAsLatentModule
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file
from gs_dataset import gs_dataset
# from gs_dataset_npy import gs_dataset 

import spconv.pytorch as spconv
from spconv.pytorch.utils import PointToVoxel
import argparse
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.utils.data as Data
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt

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

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_usage = "L1" # L1, sinkhorn, chamfer
random_permute = 0
random_rotation = 1
random_shuffle = 1

resol = 200 # depends on the input size of VAE, or the number of Gaussians per scene
data_path = f"/your/path/to/DL3DV-10K" 

dummy_image_path = "/any/scene/from/DL3DV-10K/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/gaussian_splat/"



folder_path_each = os.listdir(data_path)
# folder_path_each.remove('.ipynb_checkpoints')
num_epochs = 200000
# save_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/train_pointvae_{resol}/"  # train_vae    train_ae_only2
save_path = f"/your/save/path/"



bch_size = 200 #200
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


#### fully connected VAE
# class GS_encoder(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=56, skip=[4], output_ch=4):
#         super(GS_encoder, self).__init__()
#         self.tau = 0.5
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.skips = skip
#         self.output_ch = output_ch
#         self.pts_linears = nn.ModuleList([nn.Linear(input_ch,W)])
#         for i in range(D-1):
#             self.pts_linears.append(nn.Linear(W, W))
#             self.pts_linears.append(nn.BatchNorm1d(W))
        
#         self.output_linear_0 = nn.Linear(W, output_ch)
#         self.output_linear_1 = nn.Linear(W, output_ch)

#         self.ones_init = torch.ones([1, output_ch])
#         self.scale_0 = nn.Parameter(self.ones_init)
#         # self.scale_i = nn.Parameter(W)
#         #self.scale_1 = nn.Linear(1,output_ch)
        
#         #self.class_layer = nn.Linear(output_ch, 2)
#         self.act = nn.LeakyReLU(0.2)
       
#         self.bn_layer_output = nn.BatchNorm1d(output_ch)
#     def forward(self, x):
#         for i, l in enumerate(self.pts_linears):
#             x = self.pts_linears[i](x)
#             x = F.relu(x)

#         # x = self.bn_layer_inner(x)
#         # mu = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.scale_0)) * self.bn_layer_output(self.output_linear_0(x))
#         # log_var = torch.sqrt((1 - self.tau) * torch.sigmoid(-self.scale_0)) * self.bn_layer_output(self.output_linear_1(x))
       
#         # mu = self.bn_layer_output(self.output_linear_0(x))
#         # log_var = self.bn_layer_output(self.output_linear_1(x))
        
#         mu = self.output_linear_0(x)
#         log_var = self.output_linear_1(x)
#         #label = F.sigmoid(self.class_layer(x))
#         return mu, log_var #x, label

# class GS_decoder(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=4, skip=[4], output_ch=56):
#         super(GS_decoder, self).__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.skips = skip
#         self.output_ch = output_ch
#         self.pts_linears = nn.ModuleList([nn.Linear(input_ch,W)])

#         for i in range(D-1):
#             self.pts_linears.append(nn.Linear(W, W))
#             self.pts_linears.append(nn.BatchNorm1d(W))
        
#         self.output_linear = nn.Linear(W, output_ch)
#         self.act = nn.LeakyReLU(0.2)
#         self.bn_layer_output = nn.BatchNorm1d(output_ch)
        
#     def forward(self, x):
#         for i, l in enumerate(self.pts_linears):
#             x = self.pts_linears[i](x)
#             x = F.relu(x)
#             # x = self.act(x)
#         # x = self.bn_layer_output(self.output_linear(x))
#         x = self.output_linear(x)
#         return x

# # torch.manual_seed(0)
# # eps = torch.randn([bch_size, 16384])
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()

#         # self.encoder = GS_encoder(8,256,14,[4],14)
#         # self.decoder = GS_decoder(8,256,14,[4],14)
#         self.encoder = GS_encoder(4,256,14*resol**2,[4],16384)
#         self.decoder = GS_decoder(4,256,16384,[4],14*resol**2)
#         # self.encoder = GS_encoder(4,256,resol**2,[4],255)
#         # self.decoder = GS_decoder(4,256,255,[4],resol**2)

#     def encode(self, x):
#         return self.encoder(x)

#     def decode(self, z):
#         return self.decoder(z)

#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         std = torch.exp(0.5 * log_var).to(log_var.device)
#         eps = torch.randn_like(std).to(std.device)
#         z = mu + std * eps.to(std.device) 
#         UV_gs_recover = self.decode(z)
#         return mu, log_var, z, UV_gs_recover


######## perciever VAE
config_path_perciever = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
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



################# eval
# num_epochs = ***
# gs_autoencoder = perceiver_encoder_decoder
# # gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, os.path.join(save_path, subpath))))
# gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, str(int(num_epochs)))))
# gs_autoencoder = nn.DataParallel(perceiver_encoder_decoder)
# gs_autoencoder.to(device)
#################

optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=1e-4, betas=[0.9, 0.999])


gs_dataset = gs_dataset(data_path, resol = 128, random_permute = True, train=True)
trainDataLoader = Data.DataLoader(dataset=gs_dataset, batch_size=bch_size, shuffle=True, num_workers=12) 

gen_vxs_from_pts = PointToVoxel(vsize_xyz=[0.2, 0.2, 0.2],
                   coors_range_xyz=[-8, -8, -8, 8, 8, 8],
                   num_point_features=14,
                   max_num_voxels=10000,
                   max_num_points_per_voxel=40)

##### for PE
voxel_reso = 40
x_y = np.linspace(-8, 8, voxel_reso)   # 50
z_res = np.linspace(-8, 8, voxel_reso) # 16
xv, yv, zv = np.meshgrid(x_y, x_y, x_y, indexing='ij')
voxel_centers = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

##### for output query
output_voxel_reso = 40
output_x_y = np.linspace(-8, 8, output_voxel_reso)   
output_z_res = np.linspace(-8, 8, output_voxel_reso) 
output_xv, output_yv, output_zv = np.meshgrid(output_x_y, output_x_y, output_x_y, indexing='ij')
output_volume_centers = torch.tensor(np.vstack([output_xv.ravel(), output_yv.ravel(), output_zv.ravel()]).T, dtype=torch.float32)


volume_dims = 40
resolution = 16.0/volume_dims
origin_offset = torch.tensor(np.array([(volume_dims - 1) / 2, (volume_dims - 1) / 2, (volume_dims - 1) / 2]) * resolution, dtype=torch.float32).to(device)
    
for epoch in tqdm(range(num_epochs)):
       for i_batch, UV_gs_batch in enumerate(trainDataLoader):
           UV_gs_batch = UV_gs_batch[0].type(torch.float32).to(device)
           if epoch % 1 == 0 and random_permute == 1:
              UV_gs_batch = UV_gs_batch[:,torch.randperm(UV_gs_batch.size()[1])]
           if epoch % 5 == 0 and epoch > 1 and random_rotation ==1:
              rand_rot_comp = special_ortho_group.rvs(3)
              rand_rot = torch.tensor(np.dot(rand_rot_comp, rand_rot_comp.T), dtype = torch.float32).to(UV_gs_batch.device)
              UV_gs_batch[:,:,4:7] = UV_gs_batch[:,:,4:7]@rand_rot
              ###### for PE
              for bcbc in range(UV_gs_batch.shape[0]):
                  shifted_points = UV_gs_batch[bcbc,:,4:7] + origin_offset
                  voxel_indices = torch.floor(shifted_points / resolution)
                  voxel_indices = torch.clip(voxel_indices, 0, volume_dims - 1)
                  voxel_centers = (voxel_indices - (volume_dims - 1) / 2) * resolution
                  UV_gs_batch[bcbc,:,:3] = torch.tensor(voxel_centers, dtype=torch.float32)
                  
                  #########
                  # coord_min = np.min(np.array(UV_gs_batch[bcbc,:,1:4].cpu()), 0)
                  # coord = np.array(UV_gs_batch[bcbc,:,4:7].cpu()) - coord_min
                  # uniq_idx, count = voxelize(coord, 0.4, 'fnv') # [-8, 8] with voxel_size=0.4    # ravel, fnv
                  # UV_gs_batch[bcbc,:,3] = torch.tensor(uniq_idx, dtype=torch.float32).to(device)
                  ########
                  # voxels_tv, indices_tv, num_p_in_vx_tv, pc_voxel_id = gen_vxs_from_pts.generate_voxel_with_id(UV_gs_batch[bcbc,:,1:].cpu(),empty_mean=True)
                  # UV_gs_batch[bcbc,:,0] = torch.tensor(pc_voxel_id, dtype=torch.float32).to(device)
           
    
           loss = 0.0
           loss_render = 0.0
           optimizer.zero_grad()
       #UV_gs_batch = UV_gs_batch.reshape([len(folder_path_each), 64, 64, 56])

           ##### PE with voxel ID
           # gs_voxel_ID = []
           # for batch_id_voxel in range(UV_gs_batch.shape[0]):
           #     voxels_tv, indices_tv, num_p_in_vx_tv, pc_voxel_id = gen_vxs_from_pts.generate_voxel_with_id(UV_gs_batch[batch_id_voxel,:,:].cpu(),empty_mean=True)
           #     gs_voxel_ID.append(torch.tensor(pc_voxel_id))
           # gs_voxel_ID = torch.stack(gs_voxel_ID).to(device).unsqueeze(-1)
          
           # UV_gs_batch_voxel_id = torch.cat((gs_voxel_ID, UV_gs_batch),dim=2)
        
           shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(UV_gs_batch,UV_gs_batch,UV_gs_batch, UV_gs_batch[:,:,:3])
           
           if enable_rendering_loss == 1:
              if epoch % k_rendering_loss == 0:
                 viewpoint_stack_test = []
                 random_idx = random.sample(range(0, len(UV_gs_batch)), 2) 
                 for iik in range(len(random_idx)): 
                     idx_batch = random_idx[iik]
                     dummy_image_path = "/home/qgao/sensei-fs-link/Dataset/scripts/DL3DV-10K-Benchmark/" + folder_path_each[idx_batch] +"/gaussian_splat/"
                     model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", resol, False]
                     dataset_for_gs = group_extract(model_params_list, model_params_value)
                     gaussians = GaussianModel(dataset_for_gs.sh_degree)
                     scene = Scene(dataset_for_gs, gaussians)
                     viewpoint_stack_test.append(scene.getTrainCameras().copy())
                     training_setup_for_gs = group_extract(optimization_params_list, optimization_params_value)
                     pipe = group_extract(pipeline_params_list, pipeline_params_value)
                  
                     viewpoint = viewpoint_stack_test[0]   #[idx_batch]
                     recovered_idx = UV_gs_recover[idx_batch] #* (norm_max-norm_min) + norm_min
                     gaussians._xyz = recovered_idx[:,:3]
                     gaussians._features_dc = recovered_idx[:,3:6][:,None,:]
                     gaussians._features_rest = torch.zeros([recovered_idx.shape[0], 0, 3]).to(recovered_idx.device)
                     gaussians._opacity = recovered_idx[:, 6][:,None]
                     gaussians._scaling = recovered_idx[:, 7:10]
                     gaussians._rotation = recovered_idx[:, 10:14]
                     
                     for n_views in range(2):
                         rand_idx = randint(0, len(viewpoint)-1)
                         view_idx = viewpoint[rand_idx]
                         render_pkg = render(view_idx, gaussians, pipe, background)
                         image = render_pkg["render"]
                         gt_image = view_idx.original_image
                         loss_render += l1_loss(image, gt_image)

           KL_loss = - 0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
         
           if loss_usage == "L1":
               loss += torch.norm(UV_gs_recover.reshape(UV_gs_batch.shape[0],-1,14) - UV_gs_batch[:,:, 4:], p=2)/UV_gs_batch.shape[0] + 1e-5*KL_loss + 10*loss_render

              #### for output query setting
              # loss += torch.norm(UV_gs_recover[:,:40000,:14].reshape(UV_gs_batch.shape[0],-1,14) - UV_gs_batch[:,:,4:], p=2)/UV_gs_batch.shape[0] + 1e-5*KL_loss + 10*loss_render
           elif loss_usage == "chamfer":
              loss += torch.mean(chamferDist(UV_gs_batch.reshape([bch_size, -1, 14])[:,:,:3], UV_gs_recover.reshape([bch_size, -1, 14])[:,:,:3])) + 0.01*torch.mean(chamferDist(UV_gs_batch.reshape([bch_size, -1, 14])[:,:,3:], UV_gs_recover.reshape([bch_size, -1, 14])[:,:,3:])) + 0.0001*KL_loss + 10*loss_render
           elif loss_usage == "sinkhorn":
                sinkhorn_loss_ = sinkhorn_eff(UV_gs_batch.contiguous().reshape([bch_size, -1, 14]), UV_gs_recover.contiguous().reshape([bch_size, -1, 14])).mean()
                loss += sinkhorn_loss_ + 0.001*KL_loss + 10*loss_render
               
           if epoch % 100 == 0: 
              print(f"loss={loss.item()}  ,  kl_loss = {KL_loss.item()}")
           if epoch % 1000 == 0:
                gs_autoencoder.eval()
                # test the reconstruction quality
                recovered_1 = UV_gs_recover[0].reshape(-1,14) 
                gaussians._xyz = recovered_1[:,:3]
                gaussians._features_dc = recovered_1[:,3:6][:,None,:]
                gaussians._features_rest = torch.zeros([recovered_1.shape[0], 0, 3]).to(recovered_1.device)
                gaussians._opacity = recovered_1[:, 6][:,None]
                gaussians._scaling = recovered_1[:, 7:10]
                gaussians._rotation = recovered_1[:, 10:14]
                gaussians.save_ply(save_path+"recovered.ply")
                
                transform = T.ToPILImage()
                # number of images for visualization
                vis_num = 3
                viewpoint = viewpoint_stack[0]
                # for i_vis in range(0, vis_num):
                #   view_i = viewpoint[i_vis]
                #   render_pkg = render(view_i, gaussians, pipe, background)
                #   image = render_pkg["render"]
                #   gt_image = view_i.original_image
                #   img_recovered = transform(image)
                #   gt_image = transform(gt_image)
                #   gt_image.save(f"{save_path}gt_{i_vis}.png")
                #   img_recovered.save(f"{save_path}reco_{i_vis}.png")
                if epoch >= 10000 and epoch % 10000 == 0:
                    gs_autoencoder.train()
                    # torch.save(gs_emb, f"{save_path}gs_emb_{epoch}.pt")
                    torch.save(mu, f"{save_path}gs_mu_{epoch}_10.pt")
                    torch.save(log_var, f"{save_path}gs_var_{epoch}_10.pt")
                    torch.save(z, f"{save_path}gs_emb_{epoch}_10.pt")
                    subpath = f"{int(epoch)}_query.pth"
                    torch.save(gs_autoencoder.module.state_dict(), os.path.join(save_path, subpath))
                gs_autoencoder.train()
           loss.backward()
           optimizer.step()




# test the reconstruction quality
recovered_1 = UV_gs_recover[0].reshape(-1,14)# * (norm_max-norm_min) + norm_min
gaussians._xyz = recovered_1[:,:3]
gaussians._features_dc = recovered_1[:,3:6][:,None,:]
gaussians._features_rest = torch.zeros([recovered_1.shape[0], 0, 3]).to(recovered_1.device)
gaussians._opacity = recovered_1[:, 6][:,None]
gaussians._scaling = recovered_1[:, 7:10]
gaussians._rotation = recovered_1[:, 10:14]
gaussians.save_ply(save_path+"recovered.ply")

transform = T.ToPILImage()
# number of images for visualization
vis_num = 3
viewpoint = viewpoint_stack[0]
for i_vis in range(0, vis_num):
  view_i = viewpoint[i_vis]
  render_pkg = render(view_i, gaussians, pipe, background)
  image = render_pkg["render"]
  gt_image = view_i.original_image
  img_recovered = transform(image)
  gt_image = transform(gt_image)
  gt_image.save(f"{save_path}gt_{i_vis}.png")
  img_recovered.save(f"{save_path}reco_{i_vis}.png")
# torch.save(gs_emb, save_path+f"gs_emb_{epoch+1}.pt")
torch.save(mu, f"{save_path}gs_mu_{epoch+1}_10.pt")
torch.save(log_var, f"{save_path}gs_var_{epoch+1}_10.pt")
torch.save(z, f"{save_path}gs_emb_{epoch+1}_10.pt")
subpath = f"{int(epoch+1)}_10.pth"
torch.save(gs_autoencoder.module.state_dict(), os.path.join(save_path, subpath))

        
        
        
    
    
    
    
    