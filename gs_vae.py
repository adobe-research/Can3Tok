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
import json
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from voxelize import voxelize
import torch.utils.data as Data
from gs_dataset_original import gs_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


resol = 128 #128
random_permute = 0
random_shuffle = 1

data_path = f"/your/path/to/dl3dv/"
dummy_image_path = "/your/path/to/DL3DV-10K-Benchmark/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/gaussian_splat/"

folder_path_each = os.listdir(data_path)
# folder_path_each.remove('.ipynb_checkpoints')
num_epochs = 200000
save_path = f"/your/path/to/save/train_vae_{resol}/"  # train_vae    train_ae_only2
bch_size = 40 # 100
k_rendering_loss = 1000
enable_rendering_loss = 0
label_gt = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).to(device)
L2=torch.nn.CrossEntropyLoss()
LBCE = torch.nn.BCELoss()
class GroupParams:
    pass

def group_extract(param_list, param_value):
    group = GroupParams()
    for idx in range(len(param_list)):
        setattr(group, param_list[idx], param_value[idx])
    return group

# class CameraInfo(NamedTuple):
#     uid: int
#     R: np.array
#     T: np.array
#     FovY: np.array
#     FovX: np.array
#     image: np.array
#     image_path: str
#     image_name: str
#     width: int
#     height: int
#     scale_factor: np.array

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

# class GS_encoder(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=56, skip=[4], output_ch=4):
#         super(GS_encoder, self).__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.skips = skip
#         self.output_ch = output_ch
#         self.pts_linears = nn.ModuleList(
#             [nn.Linear(input_ch,W)] + [nn.Linear(W, W) for i in range(D-1)])
#         self.output_linear = nn.Linear(W, output_ch)
#         self.act = nn.LeakyReLU(0.1)
#     def forward(self, x):
#         for i, l in enumerate(self.pts_linears):
#             x = self.pts_linears[i](x)
#             x = F.relu(x)
#             # x = self.act(x)
#         x = self.output_linear(x)
#         return x

# class GS_decoder(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=4, skip=[4], output_ch=56):
#         super(GS_decoder, self).__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.skips = skip
#         self.output_ch = output_ch
#         self.pts_linears = nn.ModuleList(
#             [nn.Linear(input_ch,W)] + [nn.Linear(W, W) for i in range(D-1)])
#         self.output_linear = nn.Linear(W, output_ch)
#         self.act = nn.LeakyReLU(0.1)
#     def forward(self, x):
#         for i, l in enumerate(self.pts_linears):
#             x = self.pts_linears[i](x)
#             x = F.relu(x)
#             # x = self.act(x)
#         x = self.output_linear(x)
#         return x
    
    
class GS_encoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch=14, skip=[4], output_ch=4):
        super(GS_encoder, self).__init__()
        # self.D = D
        # self.W = W
        self.input_ch = input_ch
        # self.skips = skip
        self.output_ch = output_ch
        self.pts_linears = nn.ModuleList(
            [nn.Conv2d(input_ch, input_ch, 1)] + [nn.Conv2d(input_ch, input_ch, 1) for i in range(D-1)])
        #self.output_linear = nn.Conv2d(input_ch, output_ch, 3, padding=1, stride=2)
        self.output_linear = nn.Conv2d(input_ch, output_ch, 1)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = F.relu(x)
            # x = self.act(x)
        x = self.output_linear(x)
        return x

class GS_decoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch=14, skip=[4], output_ch=4):
        super(GS_decoder, self).__init__()
       # self.D = D
        # self.W = W
        self.input_ch = input_ch
        # self.skips = skip
        self.output_ch = output_ch
        #self.first_linear = nn.ConvTranspose2d(input_ch, output_ch, 2, padding=0, stride=2)
        self.first_linear = nn.ConvTranspose2d(input_ch, output_ch, 1)
        self.pts_linears = nn.ModuleList(
            [nn.Conv2d(output_ch, output_ch, 1)] + [nn.Conv2d(output_ch, output_ch, 1) for i in range(D-1)])
        # self.output_linear = nn.ConvTranspose2d(input_ch, output_ch, 3, padding=0, stride=2)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.first_linear(x)
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = F.relu(x)
            # x = self.act(x)
        return x
    
# class GS_encoder(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=56, skip=[4], output_ch=4):
#         super(GS_encoder, self).__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.skips = skip
#         self.output_ch = output_ch
#         self.pts_linears = nn.ModuleList(
#             [nn.Linear(input_ch,W)] + [nn.Linear(W, W) for i in range(D-1)])
#         self.output_linear_0 = nn.Linear(W, output_ch)
#         self.output_linear_1 = nn.Linear(W, output_ch)
#         #self.class_layer = nn.Linear(output_ch, 2)
#         self.act = nn.LeakyReLU(0.2)
#     def forward(self, x):
#         for i, l in enumerate(self.pts_linears):
#             x = self.pts_linears[i](x)
#             x = F.relu(x)
#             # x = self.act(x)
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
#         self.pts_linears = nn.ModuleList(
#             [nn.Linear(input_ch,W)] + [nn.Linear(W, W) for i in range(D-1)])
#         self.output_linear = nn.Linear(W, output_ch)
#         self.act = nn.LeakyReLU(0.2)
#     def forward(self, x):
#         for i, l in enumerate(self.pts_linears):
#             x = self.pts_linears[i](x)
#             x = F.relu(x)
#             # x = self.act(x)
#         x = self.output_linear(x)
#         return x


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
#         #x = torch.cat([x, scale_factor_each], dim = 1)
#         for i, l in enumerate(self.pts_linears):
#             x = self.pts_linears[i](x)
#             x = F.relu(x)
#             # x = self.act(x)
#         # x = self.bn_layer_output(self.output_linear(x))
#         x = self.output_linear(x)
#         return x

# torch.manual_seed(0)
# eps = torch.randn([bch_size, 16384])
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # self.encoder = GS_encoder(8,256,14,[4],14)
        # self.decoder = GS_decoder(8,256,14,[4],14)
        self.encoder = GS_encoder(4,1024,14*resol**2,[4],16384)
        self.decoder = GS_decoder(4,1024,16384,[4],14*resol**2)   # self.decoder = GS_decoder(4,256,16384+3,[4],14*resol**2)
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
        return mu, log_var, z, UV_gs_recover
    



gs_autoencoder = Network()
if torch.cuda.device_count() > 1:
  gs_autoencoder = nn.DataParallel(gs_autoencoder)
gs_autoencoder.to(device)

# gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, str(int(num_epochs)))))
optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=1e-4, betas=[0.9, 0.999])

# ##eval
# num = 150000
# which_scene = 0
# gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, str(int(num)))))
# gs_autoencoder.eval()
# gs_params_path_each = data_path + folder_path_each[which_scene] + f"/point_cloud/iteration_30000/point_cloud_{resol}_norm.ply"
# norm_max_o = torch.load(save_path+f"norm_max_{num}.pt")
# norm_min_o = torch.load(save_path+f"norm_min_{num}.pt")
# latents = torch.load(save_path+f"gs_emb_{num}.pt")
# mu = torch.load(save_path+f"gs_mu_{num}.pt").cuda()
# plydata = PlyData.read(gs_params_path_each)
# scale_factor_each = torch.load(save_path+f"scale_factor_each.pt")

# # dummy_image_path = "/home/qgao/sensei-fs-link/Dataset/scripts/DL3DV-10K-Benchmark/" + folder_path_each[which_scene] + "/gaussian_splat/"
# dummy_image_path = "/home/qgao/sensei-fs-link/Dataset/scripts/DL3DV-10K-Benchmark/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/gaussian_splat/"
# model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", 256, False]
# dataset_for_gs = group_extract(model_params_list, model_params_value)
# gaussians = GaussianModel(dataset_for_gs.sh_degree)
# scene = Scene(dataset_for_gs, gaussians)
# # train_dataset = scene.getTrainCameras()
# viewpoint_stack = []
# viewpoint_stack.append(scene.getTrainCameras().copy())
# training_setup_for_gs = group_extract(optimization_params_list, optimization_params_value)
# pipe = group_extract(pipeline_params_list, pipeline_params_value)

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
# # gs_full_params_norm = ((gs_full_params.reshape([resol, resol, 14]) - norm_min_o) / (norm_max_o - norm_min_o))
# # gs_full_params_norm_pe = torch.zeros_like(gs_full_params_norm).cuda()
# # for pe_ch in range(0,3):
# #     gs_full_params_norm_pe[:,:,pe_ch] = gs_full_params_norm[:,:,pe_ch] - gs_full_params_norm[:,:,pe_ch].mean()
# # gs_full_params_norm_pe[:,:,3:] = gs_full_params_norm_pe[:,:,3:]

# #perm_inputs = gs_full_params_norm[torch.randperm(gs_full_params_norm.size()[0])]
# # perm_inputs = gs_full_params_norm
# # perm_inputs = perm_inputs.reshape([1, -1])
# # for random row-permutation
# # mu, log_var, z, UV_gs_recover = gs_autoencoder(gs_full_params[torch.randperm(gs_full_params.size()[0])].reshape([1, -1]))


# ## reconstruction check
# dummpy_gs_input = torch.zeros([bch_size, 14*resol**2]).cuda()
# mu, log_var, z, UV_gs_recover = gs_autoencoder(dummpy_gs_input + gs_full_params.reshape([1, -1]))

# ## latent interpolation
# # mix_weight = 0.4
# # mix_latent = ((torch.randn_like(latents).to(latents.device)+mu[which_scene])*mix_weight + (torch.randn_like(latents).to(latents.device)+mu[which_scene+1])*(1-mix_weight))
# # # mix_latent = (torch.randn_like(latents).to(latents.device)+mu[which_scene] + torch.randn_like(latents).to(latents.device)+mu[which_scene+2])/2
# # UV_gs_recover = gs_autoencoder.decode(mix_latent, scale_factor_each)#[which_scene]    #(torch.randn_like(latents).to(latents.device)+mu[which_scene])[which_scene]

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


gs_dataset = gs_dataset(data_path, resol = resol, train=True)
trainDataLoader = Data.DataLoader(dataset=gs_dataset, batch_size=bch_size, shuffle=True, num_workers=12) 


for epoch in tqdm(range(num_epochs)):
       for i_batch, UV_gs_batch in enumerate(trainDataLoader):
           UV_gs_batch = UV_gs_batch[0].to(dtype = torch.float32).to(device)[:,:,1:]
       # if epoch % 500 == 0 and random_permute == 1:
       #     UV_gs = UV_gs[:,torch.randperm(UV_gs.size()[1])]
       # if epoch % 1 == 0 and random_shuffle == 1:
       #     UV_gs = UV_gs[torch.randperm(UV_gs.size()[0])]
       # loop_num = len(folder_path_each)//bch_size
       # for bch in range(0, loop_num):
       #     UV_gs_batch = UV_gs[bch*bch_size:bch*bch_size+bch_size]
           loss = 0.0
           loss_render = 0.0
           optimizer.zero_grad()
           mu, log_var, z, UV_gs_recover = gs_autoencoder(UV_gs_batch.reshape([UV_gs_batch.shape[0], -1]))
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
                     recovered_idx = UV_gs_recover[idx_batch] 
                     recovered_idx = recovered_idx.reshape(-1,14)
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
           loss += torch.norm(UV_gs_recover - UV_gs_batch.reshape([UV_gs_batch.shape[0],-1]), p=2)/len(UV_gs_batch) + 1e-5*KL_loss + 10*loss_render
           if epoch % 100 == 0: 
              print(f"loss={loss.item()}  ,  kl_loss = {KL_loss.item()}")

         
           if epoch % 1000 == 0:
                gs_autoencoder.eval()
               # test the reconstruction quality
                recovered_1 = UV_gs_recover[0] 
                recovered_1 = recovered_1.reshape(-1,14)
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
                  view_i = viewpoint[i_vis]  # i_vis
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
                    
                    torch.save(UV_gs_norm_factors, f"{save_path}norm_xyz_{epoch}.pt")
                    torch.save(gs_autoencoder.state_dict(), os.path.join(save_path, str(int(epoch))))
                gs_autoencoder.train()
           loss.backward()
           optimizer.step()




# test the reconstruction quality

recovered_1 = UV_gs_recover[0]
recovered_1 = recovered_1.reshape(-1,14)
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

torch.save(UV_gs_norm_factors, f"{save_path}norm_xyz_{epoch+1}.pt")
torch.save(gs_autoencoder.state_dict(), os.path.join(save_path, str(int(epoch+1))))
torch.save(scale_factor_each, f"{save_path}scale_factor_each.pt")

        
        
        
    
    
    
    
    