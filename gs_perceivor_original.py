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
from gs_dataset_original import gs_dataset
import argparse
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.utils.data as Data
from scipy.stats import special_ortho_group
from scipy.linalg import sqrtm

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

resol = 128 #128
data_path = f"/your/path/to/DL3DV-10K/after/3DGS/optimization" 
dummy_image_path = "/your/path/to/DL3DV-10K/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/gaussian_splat/"

folder_path_each = os.listdir(data_path)
# folder_path_each.remove('.ipynb_checkpoints')
num_epochs = 200000
save_path = f"/your/path/to/save/train_pointvae_{resol}/"  # train_vae    train_ae_only2
# save_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/filter_train_pointvae/"
bch_size = 200  #100
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


################# toy eval
ckpt = 110000
subpath = f"{ckpt}_test.pth"
gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, os.path.join(save_path, subpath))),strict=True)

gs_autoencoder.eval()
# gs_autoencoder = nn.DataParallel(gs_autoencoder)
# exit()
#################

################# eval
# num_epochs = 180000
# gs_autoencoder = perceiver_encoder_decoder
# # gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, os.path.join(save_path, subpath))))
# gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, str(int(num_epochs)))))
# gs_autoencoder = nn.DataParallel(perceiver_encoder_decoder)
# gs_autoencoder.to(device)
#################

optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=1e-4, betas=[0.9, 0.999])

# optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

# eval
# # num = 10000
# which_scene = -1
# # gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, str(int(num)))))
# gs_autoencoder.eval()
# gs_params_path_each = data_path + folder_path_each[which_scene] + f"/point_cloud/iteration_30000/point_cloud_{resol}_norm.ply"
# # gs_params_path_each = "/home/qgao/sensei-fs-link/gaussian-splatting/dl3dv_test/002afdbeb148f881f8a19e9b6e99d84fce95156085cedd37cccd768aab3eb70b"+ f"/point_cloud/iteration_30000/point_cloud_{resol}_norm.ply"
# # norm_max_o = torch.load(save_path+f"norm_max_{num}.pt")
# # norm_min_o = torch.load(save_path+f"norm_min_{num}.pt")
# # latents = torch.load(save_path+f"gs_emb_{num}.pt")
# # mu = torch.load(save_path+f"gs_mu_{num}.pt").cuda()
# plydata = PlyData.read(gs_params_path_each)
# xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                 np.asarray(plydata.elements[0]["y"]),
#                 np.asarray(plydata.elements[0]["z"])),  axis=1)
# # normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
# #                np.asarray(plydata.elements[0]["ny"]),
# #                np.asarray(plydata.elements[0]["nz"])),  axis=1)
# color_rgb = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
#                       np.asarray(plydata.elements[0]["f_dc_1"]),
#                     np.asarray(plydata.elements[0]["f_dc_2"])),  axis=1)
# opacity = np.asarray(plydata.elements[0]["opacity"])
# scale = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
#                   np.asarray(plydata.elements[0]["scale_1"]),
#                   np.asarray(plydata.elements[0]["scale_2"])),  axis=1)
# rot = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
#                 np.asarray(plydata.elements[0]["rot_1"]),
#                 np.asarray(plydata.elements[0]["rot_2"]),
#                 np.asarray(plydata.elements[0]["rot_3"])),  axis=1)
# gs_full_params = torch.tensor(np.concatenate((xyz, color_rgb, opacity[:,None], scale, rot), axis=1)).cuda()

# ## reconstruction check
# dummpy_gs_input = torch.zeros([bch_size, resol**2, 14]).cuda()
# shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(gs_full_params.reshape([1, -1, 14])+dummpy_gs_input, gs_full_params.reshape([1, -1, 14])+dummpy_gs_input,gs_full_params.reshape([1, -1, 14])+dummpy_gs_input,gs_full_params.reshape([1, -1, 14])+dummpy_gs_input)

# ## latent interpolation
# # mix_weight = 0.4
# # mix_latent = ((torch.randn_like(latents).to(latents.device)+mu[which_scene])*mix_weight + (torch.randn_like(latents).to(latents.device)+mu[which_scene+1])*(1-mix_weight))
# # # mix_latent = (torch.randn_like(latents).to(latents.device)+mu[which_scene] + torch.randn_like(latents).to(latents.device)+mu[which_scene+2])/2
# # UV_gs_recover = gs_autoencoder.decode(mix_latent)#[which_scene]    #(torch.randn_like(latents).to(latents.device)+mu[which_scene])[which_scene]

# # UV_gs_recover = gs_autoencoder.decode(torch.randn_like(latents).to(latents.device)+mu[which_scene], scale_factor_each)[which_scene] 

# # UV_gs_recover = gs_autoencoder.decode(latents.to(latents.device))[which_scene]

# # UV_gs_recover = gs_full_params.reshape([1, -1])
# # mu, std, z, UV_gs_recover = gs_autoencoder(perm_inputs)
# UV_gs_recover = UV_gs_recover[0].reshape([resol, resol, 14])# * (norm_max_o-norm_min_o) + norm_min_o
# recovered_idx = UV_gs_recover.reshape(-1,14)
# gaussians._xyz = recovered_idx[:,:3]
# gaussians._features_dc = recovered_idx[:,3:6][:,None,:]
# gaussians._features_rest = torch.zeros([recovered_idx.shape[0], 0, 3]).to(recovered_idx.device)
# gaussians._opacity = recovered_idx[:, 6][:,None]
# gaussians._scaling = recovered_idx[:, 7:10]
# gaussians._rotation = recovered_idx[:, 10:14]
# transform = T.ToPILImage()
# view_1 = viewpoint_stack[0][1]
# render_pkg = render(view_1, gaussians, pipe, background)
# image = render_pkg["render"]
# gt_image = view_1.original_image
# img_recovered = transform(image)
# gt_image = transform(gt_image)
# gt_image.save(save_path+"perm_gt.png")
# img_recovered.save(save_path+"perm_recovered.png")
# gaussians.save_ply(save_path+"recovered.ply")

# gaussians._xyz = gs_full_params[:,:3]
# gaussians._features_dc = gs_full_params[:,3:6][:,None,:]
# gaussians._features_rest = torch.zeros([gs_full_params.shape[0], 0, 3]).to(recovered_idx.device)
# gaussians._opacity = gs_full_params[:, 6][:,None]
# gaussians._scaling = gs_full_params[:, 7:10]
# gaussians._rotation = gs_full_params[:, 10:14]
# gaussians.save_ply(save_path+"gt.ply")
# exit()


#########################################
#### reconstruction check
# gs_dataset = gs_dataset(data_path, resol = 128, random_permute = False, train=True)
# trainDataLoader = Data.DataLoader(dataset=gs_dataset, batch_size=bch_size, shuffle=False, num_workers=12) 

# gs_autoencoder.eval()

# for i_batch, UV_gs_batch_raw in enumerate(trainDataLoader):
#     UV_gs_batch = UV_gs_batch_raw[0].to(dtype = torch.float32).to(device)
#     rand_rot_comp = special_ortho_group.rvs(3)
#     rand_rot = torch.tensor(np.dot(rand_rot_comp, rand_rot_comp.T), dtype = torch.float32).to(UV_gs_batch.device)
#     gs_params_object_test_path = f"/your/path/to/dl3dv/036b0f4b8070789373dd08617539756187ed9f64bbbb6c17cfa5277815714579//point_cloud/iteration_30000/point_cloud_128_norm.ply"
#     plydata = PlyData.read(gs_params_object_test_path)
#     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                     np.asarray(plydata.elements[0]["y"]),
#                     np.asarray(plydata.elements[0]["z"])),  axis=1)
        
#     color_rgb = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
#                           np.asarray(plydata.elements[0]["f_dc_1"]),
#                           np.asarray(plydata.elements[0]["f_dc_2"])),  axis=1)
        
#     opacity = np.asarray(plydata.elements[0]["opacity"])
        
#     scale = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
#                       np.asarray(plydata.elements[0]["scale_1"]),
#                       np.asarray(plydata.elements[0]["scale_2"])),  axis=1)
        
#     rot = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
#                     np.asarray(plydata.elements[0]["rot_1"]),
#                     np.asarray(plydata.elements[0]["rot_2"]),
#                     np.asarray(plydata.elements[0]["rot_3"])),  axis=1) 
#     UV_gs_batch = torch.zeros([UV_gs_batch.shape[0], UV_gs_batch.shape[1],18]).to(device)
#     inputs = torch.tensor(np.concatenate((np.zeros_like(opacity[:,None]),xyz, color_rgb, opacity[:,None], scale, rot), axis=1)).type(torch.FloatTensor).to(device).unsqueeze(0)
    
#     UV_gs_batch[:,:,3:] = inputs
#     ############
#     test_idx = 11
#     shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(UV_gs_batch,UV_gs_batch,UV_gs_batch,UV_gs_batch)
#     UV_gs_recover_save = UV_gs_recover[test_idx].reshape([resol, resol, 14])  # 1 is the statue
#     recovered_idx = UV_gs_recover_save.reshape(-1,14)
#     gaussians._xyz = recovered_idx[:,:3]
#     gaussians._features_dc = recovered_idx[:,3:6][:,None,:]
#     gaussians._features_rest = torch.zeros([recovered_idx.shape[0], 0, 3]).to(recovered_idx.device)
#     gaussians._opacity = recovered_idx[:, 6][:,None]
#     gaussians._scaling = recovered_idx[:, 7:10]
#     gaussians._rotation = recovered_idx[:, 10:14]
#     transform = T.ToPILImage()
#     view_1 = viewpoint_stack[0][1]
#     render_pkg = render(view_1, gaussians, pipe, background)

#     print(f"test_error = {torch.norm(UV_gs_recover.reshape(UV_gs_batch.shape[0],-1,14)[test_idx] - UV_gs_batch[test_idx,:,4:], p=2)}"))
#     gaussians.save_ply(f"/your/path/to/save/your/test/result/gaussian-splatting/dl3dv_test/test_recover.ply")
#     exit()
# exit()
#########################################


#########################################
### save each scene into latent
# gs_dataset = gs_dataset(data_path, resol = 128, random_permute = False, train=True)
# trainDataLoader = Data.DataLoader(dataset=gs_dataset, batch_size=bch_size, shuffle=False, num_workers=12) 

# gs_autoencoder.eval()
# store_list = []

# for i_batch, UV_gs_batch_raw in enumerate(trainDataLoader):
#     UV_gs_batch = UV_gs_batch_raw[0].to(device)
#     import time
#     start_time = time.time()
#     shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(UV_gs_batch,UV_gs_batch,UV_gs_batch,UV_gs_batch)
#     end_time = time.time()
#     print(f'time={(end_time-start_time)}')
#     exit()
#     store_list.append(np.array(torch.cat((z.cpu(), mu.cpu(), UV_gs_batch_raw[1].unsqueeze(-1)), dim=1).detach())) 
#     # z:[100, 16384], mu:[100, 16384], UV_gs_batch_raw[1]:[100,1]

# store_tensor = np.concatenate(store_list,axis=0)
# # np.save(f"{save_path}latent_label_DiT.npy", store_tensor)
# # np.save(f"dl3dv_test/image2latent.npy", store_tensor)

# exit()
#########################################


#########################################
#### decode z back to scene

# # def calculate_fid(real_features, generated_features):
# #     # Calculate mean and covariance of real and generated features
# #     mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
# #     mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

# #     # Calculate the squared difference of means
# #     diff = mu1 - mu2
# #     mean_diff = np.sum(diff ** 2)

# #     # Calculate sqrt of the product of covariances
# #     covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

# #     # Handle imaginary numbers in covmean (may happen due to numerical instability)
# #     if np.iscomplexobj(covmean):
# #         covmean = covmean.real

# #     # Calculate FID score
# #     fid = mean_diff + np.trace(sigma1 + sigma2 - 2 * covmean)
# #     return fid
# dit_output = torch.tensor(np.load(f"/home/qgao/sensei-fs-link/DiT/dit_output_interp.npy")).to(device)
# gt_output = torch.tensor(np.load(f"{save_path}latent_label_DiT.npy")).to(device)
# # import pdb;pdb.set_trace()
# # fid_score = calculate_fid(np.array(gt_output[:20,:16384].reshape([dit_output.shape[0],-1])), np.array(dit_output.reshape([dit_output.shape[0],-1])))
# # print(f"FID Score: {fid_score}")
# # exit()


# gs_autoencoder.eval()
# UV_gs_recover = gs_autoencoder.decode(dit_output.reshape([dit_output.shape[0],512,32]),dit_output)
# UV_gs_recover_gt = gs_autoencoder.decode(gt_output[:,:16384].reshape([gt_output.shape[0],512,32]),dit_output)
# for k_reco in range(UV_gs_recover.shape[0]):
#     recovered_1 = UV_gs_recover[k_reco].reshape(-1,14) 
#     gaussians._xyz = recovered_1[:,:3]
#     gaussians._features_dc = recovered_1[:,3:6][:,None,:]
#     gaussians._features_rest = torch.zeros([recovered_1.shape[0], 0, 3]).to(recovered_1.device)
#     gaussians._opacity = recovered_1[:, 6][:,None]
#     gaussians._scaling = recovered_1[:, 7:10]
#     gaussians._rotation = recovered_1[:, 10:14]
#     gaussians.save_ply(f"/home/qgao/sensei-fs-link/DiT/recovered_{k_reco}.ply")

#     gt_1 = UV_gs_recover_gt[k_reco].reshape(-1,14) 
#     gaussians._xyz = gt_1[:,:3]
#     gaussians._features_dc = gt_1[:,3:6][:,None,:]
#     gaussians._features_rest = torch.zeros([gt_1.shape[0], 0, 3]).to(gt_1.device)
#     gaussians._opacity = gt_1[:, 6][:,None]
#     gaussians._scaling = gt_1[:, 7:10]
#     gaussians._rotation = gt_1[:, 10:14]
#     gaussians.save_ply(f"/home/qgao/sensei-fs-link/DiT/gt_{k_reco}.ply")
    
# exit()
#########################################


gs_dataset = gs_dataset(data_path, resol = 128, random_permute = True, train=True)
trainDataLoader = Data.DataLoader(dataset=gs_dataset, batch_size=bch_size, shuffle=True, num_workers=12) 
voxel_reso = 40
x_y = np.linspace(-8, 8, voxel_reso)   # 50
z_res = np.linspace(-8, 8, voxel_reso) # 16
xv, yv, zv = np.meshgrid(x_y, x_y, x_y, indexing='ij')
volume_centers = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T


volume_dims = 40
resolution = 16.0/volume_dims
origin_offset = torch.tensor(np.array([(volume_dims - 1) / 2, (volume_dims - 1) / 2, (volume_dims - 1) / 2]) * resolution, dtype=torch.float32).to(device)


# for epoch in tqdm(range(ckpt,num_epochs)):
for epoch in tqdm(range(num_epochs)):
       for i_batch, UV_gs_batch in enumerate(trainDataLoader):
           UV_gs_batch = UV_gs_batch[0].to(dtype = torch.float32).to(device)
           if epoch % 1 == 0 and random_permute == 1:
              UV_gs_batch = UV_gs_batch[:,torch.randperm(UV_gs_batch.size()[1])]
           if epoch % 10 == 0 and random_rotation ==1:
              rand_rot_comp = special_ortho_group.rvs(3)
              # print(rand_rot_comp)
              rand_rot = torch.tensor(np.dot(rand_rot_comp, rand_rot_comp.T), dtype = torch.float32).to(UV_gs_batch.device)
              UV_gs_batch[:,:,4:7] = UV_gs_batch[:,:,4:7]@rand_rot
              ###### for PE
              for bcbc in range(UV_gs_batch.shape[0]):
                  shifted_points = UV_gs_batch[bcbc,:,4:7] + origin_offset
                  voxel_indices = torch.floor(shifted_points / resolution)
                  voxel_indices = torch.clip(voxel_indices, 0, volume_dims - 1)
                  voxel_centers = (voxel_indices - (volume_dims - 1) / 2) * resolution
                  UV_gs_batch[bcbc,:,:3] = torch.tensor(voxel_centers, dtype=torch.float32)

              
    
           loss = 0.0
           loss_render = 0.0
           optimizer.zero_grad()
       
           shape_embed, mu, log_var, z, UV_gs_recover = gs_autoencoder(UV_gs_batch,UV_gs_batch,UV_gs_batch,UV_gs_batch[:,:,:3])
           
           if enable_rendering_loss == 1:
              if epoch % k_rendering_loss == 0: #and epoch >= k_rendering_loss:
                 # for idx_batch in range(len(UV_gs_batch)):
                 viewpoint_stack_test = []
                 random_idx = random.sample(range(0, len(UV_gs_batch)), 2) 
                 for iik in range(len(random_idx)): 
                     idx_batch = random_idx[iik]
                     dummy_image_path = "/your/path/to/DL3DV-10K/" + folder_path_each[idx_batch] +"/gaussian_splat/"
                     model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", resol, False]
                     dataset_for_gs = group_extract(model_params_list, model_params_value)
                     gaussians = GaussianModel(dataset_for_gs.sh_degree)
                     scene = Scene(dataset_for_gs, gaussians)
                     # train_dataset = scene.getTrainCameras()
                     viewpoint_stack_test.append(scene.getTrainCameras().copy())
                     training_setup_for_gs = group_extract(optimization_params_list, optimization_params_value)
                     pipe = group_extract(pipeline_params_list, pipeline_params_value)
                  
                     viewpoint = viewpoint_stack_test[0] 
                     recovered_idx = UV_gs_recover[idx_batch] 
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
              loss += torch.norm(UV_gs_recover.reshape(UV_gs_batch.shape[0],-1,14) - UV_gs_batch[:,:,4:], p=2)/UV_gs_batch.shape[0] + 1e-5*KL_loss + 10*loss_render
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
                # if epoch % 10000 == 0:
                    gs_autoencoder.train()
                    # torch.save(gs_emb, f"{save_path}gs_emb_{epoch}.pt")
                    torch.save(mu, f"{save_path}gs_mu_{epoch}.pt")
                    torch.save(log_var, f"{save_path}gs_var_{epoch}.pt")
                    torch.save(z, f"{save_path}gs_emb_{epoch}.pt")
                    subpath = f"{int(epoch)}_wo_norm.pth"
                    torch.save(gs_autoencoder.module.state_dict(), os.path.join(save_path, subpath))
                gs_autoencoder.train()
           loss.backward()
           optimizer.step()




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
torch.save(mu, f"{save_path}gs_mu_{epoch+1}.pt")
torch.save(log_var, f"{save_path}gs_var_{epoch+1}.pt")
torch.save(z, f"{save_path}gs_emb_{epoch+1}.pt")
subpath = f"{int(epoch+1)}.pth"
torch.save(gs_autoencoder.module.state_dict(), os.path.join(save_path, subpath))

        
        
        
    
    
    
    
    