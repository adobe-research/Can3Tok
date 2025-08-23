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

loss_usage = "L1" # L1, sinkhorn, chamfer
random_permute = 0
random_shuffle = 1

resol = 128 #128
data_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/dl3dv-1k/"
dummy_image_path = "/home/qgao/sensei-fs-link/Dataset/scripts/DL3DV-10K-Benchmark/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/gaussian_splat/"

folder_path_each = os.listdir(data_path)
# folder_path_each.remove('.ipynb_checkpoints')
num_epochs = 200000
save_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/train_pointvae_{resol}/"  # train_vae    train_ae_only2
bch_size = 100 #111
k_rendering_loss = 1000
enable_rendering_loss = 0
label_gt = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).cuda()
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

# dataset_for_gs = group_extract(model_params_list, model_params_value)
# gaussians = GaussianModel(dataset_for_gs.sh_degree)
# scene = Scene(dataset_for_gs, gaussians)
# # train_dataset = scene.getTrainCameras()
# viewpoint_stack = scene.getTrainCameras().copy()
# training_setup_for_gs = group_extract(optimization_params_list, optimization_params_value)
# pipe = group_extract(pipeline_params_list, pipeline_params_value)



background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")


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
    
######## pointnet VAE
class GS_encoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch=56, skip=[4], output_ch=4):
        super(GS_encoder, self).__init__()
        self.tau = 0.5
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skip
        self.output_ch = output_ch
        # self.pts_linears = nn.ModuleList([nn.Linear(input_ch,W)])
        # for i in range(D-1):
        #     self.pts_linears.append(nn.Linear(W, W))
        #     self.pts_linears.append(nn.BatchNorm1d(W))
        

        # self.ones_init = torch.ones([1, output_ch])
        # self.scale_0 = nn.Parameter(self.ones_init)
        
        # self.scale_i = nn.Parameter(W)
        #self.scale_1 = nn.Linear(1,output_ch)
        
        #self.class_layer = nn.Linear(output_ch, 2)
        self.act = nn.LeakyReLU(0.2)
       
        self.bn_layer_output = nn.BatchNorm1d(output_ch)
        # self.bn_layer_output = nn.LayerNorm(output_ch)

        # self.conv_linears = nn.ModuleList([nn.Conv1d(in_channels=input_ch, out_channels=W, kernel_size=1)])
        self.conv_linears = nn.ModuleList([nn.Conv2d(in_channels=14, out_channels=10, kernel_size=3, stride=1, padding=1)])
        self.conv_linears.append(nn.BatchNorm2d(10))
        self.conv_linears.append(nn.ReLU())
            # self.conv_linears.append(nn.Conv1d(in_channels=W, out_channels=W, kernel_size=1))
        self.conv_linears.append(nn.Conv2d(in_channels=10, out_channels=6, kernel_size=3, stride=2, padding=1))
        self.conv_linears.append(nn.BatchNorm2d(6))
        self.conv_linears.append(nn.ReLU())



        
        # self.conv_linears.append(nn.Conv1d(in_channels=W, out_channels=2*W, kernel_size=1))
        self.conv_linears.append(nn.Conv2d(in_channels=6, out_channels=4, kernel_size=1))
        
        self.conv_linears.append(nn.BatchNorm2d(4))
        # self.conv_linears.append(nn.LayerNorm(2*W*14))
        # self.conv_linears.append(nn.InstanceNorm1d(2*W))
        self.conv_linears.append(nn.ReLU())
        
    
         
        self.output_linear_0 = nn.Linear(4*W, output_ch)
        self.output_linear_1 = nn.Linear(4*W, output_ch)

        self.linear_mapping = nn.Linear(16384, 4*W)
    
    def forward(self, x):
        x=x.reshape([x.shape[0], 128,128, 14]).permute(0,3,1,2)
        for i, l_layer in enumerate(self.conv_linears): 
            x = self.conv_linears[i](x)
     

        x=x.permute(0,2,3,1).reshape(x.shape[0],-1)
        x = self.linear_mapping(x)

        # x = self.bn_layer_inner(x)
        # mu = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.scale_0)) * self.bn_layer_output(self.output_linear_0(x))
        # log_var = torch.sqrt((1 - self.tau) * torch.sigmoid(-self.scale_0)) * self.bn_layer_output(self.output_linear_1(x))
        
        mu = self.bn_layer_output(self.output_linear_0(x))
        log_var = self.bn_layer_output(self.output_linear_1(x))
    
        return mu, log_var #x, label

# ####### fully-connected mlp decoder
class GS_decoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch=4, skip=[4], output_ch=56):
        super(GS_decoder, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skip
        self.output_ch = output_ch
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch,W)])
        for i in range(D-1):
            self.pts_linears.append(nn.Linear(W, W))
            self.pts_linears.append(nn.BatchNorm1d(W))
            # self.pts_linears.append(nn.LayerNorm(W))
            # self.pts_linears.append(nn.InstanceNorm1d(W))
            self.pts_linears.append(nn.ReLU())
        
        self.output_linear = nn.Linear(in_features=W, out_features=output_ch)
        # self.bn_layer_output = nn.BatchNorm1d(output_ch)
        
    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            # x = F.relu(x)
            # x = self.act(x)
        # x = self.bn_layer_output(self.output_linear(x))
  
        x = self.output_linear(x)
        return x

####### conv decoder
# class GS_decoder(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=14, skip=[4], output_ch=4):
#         super(GS_decoder, self).__init__()
#        # self.D = D
#         # self.W = W
#         self.input_ch = input_ch
#         # self.skips = skip
#         self.output_ch = output_ch
#         self.first_linear = nn.Linear(input_ch, 2048)
        
#         self.conv_0 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2)
#         self.bn_0 = nn.BatchNorm2d(512)
#         self.conv_1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=3)
#         self.bn_1 = nn.BatchNorm2d(256)
#         self.conv_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2)
#         self.bn_2 = nn.BatchNorm2d(256)
#         self.conv_3 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=4)
#         self.bn_3 = nn.BatchNorm2d(128)
#         self.conv_4 = nn.ConvTranspose2d(128, 14, kernel_size=4, stride=1)
#         self.bn_4 = nn.BatchNorm2d(14)
        
#         # self.pts_conv = nn.ModuleList(
#         #     [nn.Conv2d(output_ch, output_ch, 1)] + [nn.Conv2d(output_ch, output_ch, 1) for i in range(D-1)])
#         # self.output_linear = nn.ConvTranspose2d(input_ch, output_ch, 3, padding=0, stride=2)
#         self.act = nn.LeakyReLU(0.1)
#     def forward(self, x):
#         x = self.first_linear(x)
#         x = x.reshape([x.shape[0], -1, 2, 2])
    
#         x = F.relu(self.bn_0(self.conv_0(x)))
#         x = F.relu(self.bn_1(self.conv_1(x)))
#         x = F.relu(self.bn_2(self.conv_2(x)))
#         x = F.relu(self.bn_3(self.conv_3(x)))
#         x = self.conv_4(x)
       
#         x = x.permute(0,2,3,1)
#         x = x.reshape([x.shape[0],-1, 14])
    
#         return x.reshape([x.shape[0],-1, 14])



# torch.manual_seed(0)
# eps = torch.randn([bch_size, 16384])
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # self.encoder = GS_encoder(8,256,14,[4],14)
        # self.decoder = GS_decoder(8,256,14,[4],14)
        self.encoder = GS_encoder(4,1024,resol**2,[4],16384) # 64*64*4=16384 256-512
        self.decoder = GS_decoder(6,1024,16384,[4],14*resol**2)
        # self.encoder = GS_encoder(4,256,resol**2,[4],255)
        # self.decoder = GS_decoder(4,256,255,[4],resol**2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var).to(log_var.device)
        eps = torch.randn_like(std).to(std.device)
        z = mu + std * eps.to(std.device) 
        UV_gs_recover = self.decode(z)
        # return mu, log_var, mu, UV_gs_recover
        return mu, log_var, z, UV_gs_recover





gs_autoencoder = Network().cuda() 
# gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, str(int(num_epochs)))))
optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=2e-4, betas=[0.9, 0.999])

# eval
# num = 100000
# which_scene = 0
# gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, str(int(num)))))
# gs_autoencoder.eval()
# gs_params_path_each = data_path + folder_path_each[which_scene] + f"/point_cloud/iteration_30000/point_cloud_{resol}_norm.ply"
# norm_max_o = torch.load(save_path+f"norm_max_{num}.pt")
# norm_min_o = torch.load(save_path+f"norm_min_{num}.pt")
# latents = torch.load(save_path+f"gs_emb_{num}.pt")
# mu = torch.load(save_path+f"gs_mu_{num}.pt").cuda()
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
# mu, log_var, z, UV_gs_recover = gs_autoencoder(gs_full_params.reshape([1, -1, 14]) + dummpy_gs_input)

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


# gs_params_path_each = data_path + folder_path_each[1] + "/point_cloud/iteration_30000/point_cloud.ply"
# gaussians.load_ply(gs_params_path_each)
# import pdb;pdb.set_trace()
# folder_path_each = folder_path_each[:2]
UV_gs = []
UV_gs_norm_factors = []
UV_gs_scale_norm = []
folder_path_each = folder_path_each[:1000]
for i in range(len(folder_path_each)):
    gs_params_path_each = data_path + folder_path_each[i] + f"/point_cloud/iteration_30000/point_cloud_{resol}_norm.ply"
    # UV_gs_reshape = gaussians.load_ply(gs_params_path_each)
    plydata = PlyData.read(gs_params_path_each)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    xyz_norm_fact = np.array([[xyz[:,0].min(), xyz[:,1].min(), xyz[:,2].min()],[xyz[:,0].max(), xyz[:,1].max(), xyz[:,2].max()]])
    xyz_norm = (xyz - xyz_norm_fact[0])/(xyz_norm_fact[1] - xyz_norm_fact[0])
    
    # normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
    #                np.asarray(plydata.elements[0]["ny"]),
    #                np.asarray(plydata.elements[0]["nz"])),  axis=1)
    color_rgb = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                          np.asarray(plydata.elements[0]["f_dc_1"]),
                          np.asarray(plydata.elements[0]["f_dc_2"])),  axis=1)
    rgb_norm_fact = np.array([[color_rgb[:,0].min(), color_rgb[:,1].min(), color_rgb[:,2].min()],[color_rgb[:,0].max(), color_rgb[:,1].max(), color_rgb[:,2].max()]])
    color_rgb_norm = (color_rgb - rgb_norm_fact[0])/(rgb_norm_fact[1] - rgb_norm_fact[0])
    
    opacity = np.asarray(plydata.elements[0]["opacity"])
    opacity_norm_fact = np.array([[opacity.min()],[opacity.max()]])
    opacity_norm = (opacity - opacity_norm_fact[0])/(opacity_norm_fact[1] - opacity_norm_fact[0])
    
    scale = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                      np.asarray(plydata.elements[0]["scale_1"]),
                      np.asarray(plydata.elements[0]["scale_2"])),  axis=1)
    scale_norm_fact = np.array([[scale[:,0].min(), scale[:,1].min(), scale[:,2].min()],[scale[:,0].max(), scale[:,1].max(), scale[:,2].max()]])
    scale_norm = (scale - scale_norm_fact[0])/(scale_norm_fact[1] - scale_norm_fact[0])
    
   
    rot = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                    np.asarray(plydata.elements[0]["rot_1"]),
                    np.asarray(plydata.elements[0]["rot_2"]),
                    np.asarray(plydata.elements[0]["rot_3"])),  axis=1)
    rot_norm_fact = np.array([[rot[:,0].min(), rot[:,1].min(), rot[:,2].min(), rot[:,3].min()],[rot[:,0].max(), rot[:,1].max(), rot[:,2].max(), rot[:,3].max()]])
    rot_norm = (rot - rot_norm_fact[0])/(rot_norm_fact[1] - rot_norm_fact[0])
    

    ##### PE based on xyz
    coord_min = np.min(xyz, 0)
    coord = xyz - coord_min
    uniq_idx, count = voxelize(coord, 0.01, 'fnv')  # ravel, fnv
    
    
    # gs_full_params_norm = np.concatenate((xyz_norm, color_rgb_norm, opacity_norm[:,None], scale_norm, rot_norm), axis=1)
    gs_full_params = np.concatenate((xyz, color_rgb, opacity[:,None], scale, rot), axis=1) ##########################
    UV_gs.append(gs_full_params[uniq_idx])
    
    gs_norm_factor = np.concatenate((xyz_norm_fact, rgb_norm_fact, opacity_norm_fact, scale_norm_fact, rot_norm_fact), axis=1)
    UV_gs_norm_factors.append(gs_norm_factor)
    
UV_gs = torch.tensor(np.stack(UV_gs, axis=0)).cuda() #.reshape([len(folder_path_each), 64, 64, 56])
#UV_gs_xyz_norm = torch.tensor(np.stack(UV_gs_xyz_norm, axis=0)).cuda()
UV_gs_norm_factors = torch.tensor(np.stack(UV_gs_norm_factors, axis=0)).cuda()
norm_max = torch.zeros([UV_gs.shape[1], UV_gs.shape[2]]).cuda()
norm_min = torch.zeros([UV_gs.shape[1], UV_gs.shape[2]]).cuda()


# for ch in range(UV_gs_batch.shape[-1]):
#     norm_max[:,:,ch] = UV_gs_batch[:,:,:,ch].max()
#     norm_min[:,:,ch] = UV_gs_batch[:,:,:,ch].min() 
#     UV_gs_batch[:,:,:,ch] = (UV_gs_batch[:,:,:,ch] - norm_min[:,:,ch]) / (norm_max[:,:,ch] - norm_min[:,:,ch])
    
for epoch in tqdm(range(num_epochs)):
       if epoch % 2 == 0 and random_permute == 1:
           UV_gs = UV_gs[:,torch.randperm(UV_gs.size()[1])]
       if epoch % 1 == 0 and random_shuffle == 1:
           UV_gs = UV_gs[torch.randperm(UV_gs.size()[0])]
       loop_num = len(folder_path_each)//bch_size
       for bch in range(0, loop_num):
           UV_gs_batch = UV_gs[bch*bch_size:bch*bch_size+bch_size]
           loss = 0.0
           loss_render = 0.0
           optimizer.zero_grad()
       #UV_gs_batch = UV_gs_batch.reshape([len(folder_path_each), 64, 64, 56])

       # random permutation
       # UV_gs_batch = UV_gs_batch.reshape([UV_gs_batch.shape[0], -1, 14])
       # if epoch % 5000 == 0:
       #    UV_gs_batch = UV_gs_batch[:,torch.randperm(UV_gs_batch.size()[1]),:,:] 
       # gs_emb, UV_gs_recover = gs_autoencoder(UV_gs_batch.reshape([UV_gs_batch.shape[0], -1]))
       # UV_gs_recover = UV_gs_recover.reshape([UV_gs_batch.shape[0], 128, 128, 14]) 
       # ## rendering loss

       #### for PE
       # UV_gs_batch_pe = torch.zeros_like(UV_gs_batch).cuda()
       # for pe_ch in range(0,3):
       #     UV_gs_batch_pe[:,:,:,pe_ch] = UV_gs_batch[:,:,:,pe_ch] - UV_gs_batch[:,:,:,pe_ch].mean()
       # UV_gs_batch_pe[:,:,:,3:] = UV_gs_batch[:,:,:,3:]
       # gs_emb, UV_gs_recover = gs_autoencoder(UV_gs_batch_pe.reshape([UV_gs_batch_pe.shape[0], -1]))
           
           # mu, log_var, UV_gs_recover = gs_autoencoder(UV_gs_batch.reshape([UV_gs_batch.shape[0], -1, 14]).permute(0,2,1))
           # UV_gs_recover = UV_gs_recover.permute(0,2,1).reshape([UV_gs_batch.shape[0], resol, resol, 14])
           mu, log_var, z, UV_gs_recover = gs_autoencoder(UV_gs_batch)
           #UV_gs_recover = UV_gs_recover.reshape([UV_gs_batch.shape[0], resol, resol, 14]) 
           
           if enable_rendering_loss == 1:
              if epoch % k_rendering_loss == 0: #and epoch >= k_rendering_loss:
                 # for idx_batch in range(len(UV_gs_batch)):
                 viewpoint_stack_test = []
                 random_idx = random.sample(range(0, len(UV_gs_batch)), 2) 
                 for iik in range(len(random_idx)): 
                     idx_batch = random_idx[iik]
                     dummy_image_path = "/home/qgao/sensei-fs-link/Dataset/scripts/DL3DV-10K-Benchmark/" + folder_path_each[idx_batch] +"/gaussian_splat/"
                     model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", resol, False]
                     dataset_for_gs = group_extract(model_params_list, model_params_value)
                     gaussians = GaussianModel(dataset_for_gs.sh_degree)
                     scene = Scene(dataset_for_gs, gaussians)
                     # train_dataset = scene.getTrainCameras()
                     viewpoint_stack_test.append(scene.getTrainCameras().copy())
                     training_setup_for_gs = group_extract(optimization_params_list, optimization_params_value)
                     pipe = group_extract(pipeline_params_list, pipeline_params_value)
                  
                     viewpoint = viewpoint_stack_test[0]   #[idx_batch]
                     recovered_idx = UV_gs_recover[idx_batch] #* (norm_max-norm_min) + norm_min
                     # recovered_idx = UV_gs_recover[idx_batch].reshape([resol,resol,14]) * norm_std + norm_mean
                     # recovered_idx = UV_gs_recover[idx_batch].reshape([resol,resol,14])
                     # recovered_idx = recovered_idx * (UV_gs_norm_factors[idx_batch][1] -  UV_gs_norm_factors[idx_batch][0]) + UV_gs_norm_factors[idx_batch][0]
                     # recovered_idx = recovered_idx.reshape(-1,14)
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
                        #  loss = loss_render
                        #  loss.backward()
                        #  for name, parms in gs_autoencoder.named_parameters():	
  		                    #   print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		                      # ' -->grad_value:',parms.grad)
                        #  import pdb;pdb.set_trace()
        
           #loss += LBCE(F.sigmoid(UV_gs_recover),F.sigmoid(UV_gs_batch)) - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
           #gs_emb, UV_gs_recover = gs_autoencoder(UV_gs_batch)
           #loss += (torch.norm(UV_gs_batch - UV_gs_recover, p=2) + loss_render)/len(UV_gs_batch) 
           # KL loss + distri loss

           KL_loss = - 0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

           if loss_usage == "L1":
              loss += torch.norm(UV_gs_recover.reshape(bch_size,-1,14) - UV_gs_batch, p=2)/len(UV_gs_batch) + 0.0000000001*KL_loss + 10*loss_render
           elif loss_usage == "chamfer":
              loss += torch.mean(chamferDist(UV_gs_batch.reshape([bch_size, -1, 14])[:,:,:3], UV_gs_recover.reshape([bch_size, -1, 14])[:,:,:3])) + 0.01*torch.mean(chamferDist(UV_gs_batch.reshape([bch_size, -1, 14])[:,:,3:], UV_gs_recover.reshape([bch_size, -1, 14])[:,:,3:])) + 0.0001*KL_loss + 10*loss_render
              # loss += torch.mean(chamferDist(UV_gs_batch.reshape([bch_size, -1, 14]), UV_gs_recover.reshape([bch_size, -1, 14]),bidirectional=True)) + 0.001*KL_loss + 10*loss_render
           elif loss_usage == "sinkhorn":
               # for bsize in range(bch_size):
                   # loss += sinkhorn(UV_gs_batch.reshape([bch_size, -1, 14])[bsize], UV_gs_recover.reshape([bch_size, -1, 14])[bsize])[0]
               # loss += 0.001*KL_loss + 10*loss_render
                
                sinkhorn_loss_ = sinkhorn_eff(UV_gs_batch.contiguous().reshape([bch_size, -1, 14]), UV_gs_recover.contiguous().reshape([bch_size, -1, 14])).mean()
                loss += sinkhorn_loss_ + 0.001*KL_loss + 10*loss_render
               


           
           # loss += F.mse_loss(UV_gs_recover, UV_gs_batch) - 0.1*0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
     
           #loss += L2(label_gt, label)/len(UV_gs_batch) 
           #loss += (torch.norm(UV_gs_batch - UV_gs_recover, p=2))/len(UV_gs_batch)
           # loss += L2(UV_gs_batch, UV_gs_recover)
           if epoch % 100 == 0: 
              print(f"loss={loss.item()}  ,  kl_loss = {KL_loss.item()}")
           if epoch % 1000 == 0:
                gs_autoencoder.eval()
               # test the reconstruction quality
                recovered_1 = UV_gs_recover[0].reshape(-1,14) #* (norm_max-norm_min) + norm_min
                # recovered_1 = UV_gs_recover[0].reshape([resol,resol,14]) * norm_std + norm_mean
                # recovered_1 = UV_gs_recover[0]
                # recovered_1 = recovered_1
                # recovered_1 = recovered_1 * (UV_gs_norm_factors[0][1] -  UV_gs_norm_factors[0][0]) + UV_gs_norm_factors[0][0]
                # gaussians.load_ply(gs_params_path_each)
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
                if epoch >= 100000 and epoch % 10000 == 0:
                    # torch.save(gs_emb, f"{save_path}gs_emb_{epoch}.pt")
                    torch.save(mu, f"{save_path}gs_mu_{epoch}.pt")
                    torch.save(log_var, f"{save_path}gs_var_{epoch}.pt")
                    torch.save(z, f"{save_path}gs_emb_{epoch}.pt")
                    # torch.save(norm_mean, f"{save_path}norm_mean_{epoch}.pt")
                    # torch.save(norm_std, f"{save_path}norm_std_{epoch}.pt")
                    torch.save(norm_min, f"{save_path}norm_min_{epoch}.pt")
                    torch.save(norm_max, f"{save_path}norm_max_{epoch}.pt")
                    
                    torch.save(UV_gs_norm_factors, f"{save_path}norm_xyz_{epoch}.pt")
                    torch.save(gs_autoencoder.state_dict(), os.path.join(save_path, str(int(epoch))))
                gs_autoencoder.train()
           loss.backward()
           optimizer.step()




# test the reconstruction quality
# recovered_1 = UV_gs_recover[0] * (UV_gs_norm_factors[0][1] -  UV_gs_norm_factors[0][0]) + UV_gs_norm_factors[0][0]
recovered_1 = UV_gs_recover[0].reshape(-1,14)# * (norm_max-norm_min) + norm_min
# recovered_1 = UV_gs_recover[1]
# recovered_1 = recovered_1.reshape(-1,14)
# gaussians.load_ply(gs_params_path_each)
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

torch.save(norm_min, f"{save_path}norm_min_{epoch+1}.pt")
torch.save(norm_max, f"{save_path}norm_max_{epoch+1}.pt")
# torch.save(norm_mean, f"{save_path}norm_mean_{epoch}.pt")
# torch.save(norm_std, f"{save_path}norm_std_{epoch}.pt")
torch.save(UV_gs_norm_factors, f"{save_path}norm_xyz_{epoch+1}.pt")
torch.save(gs_autoencoder.state_dict(), os.path.join(save_path, str(int(epoch+1))))

        
        
        
    
    
    
    
    