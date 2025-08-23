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

loss_usage = "L1" # L1, sinkhorn, chamfer
random_permute = 0

resol = 128 #200, 316...
data_path = f"/your/path/to/DL3DV-10K/after/3DGS/optimization/gaussian-splatting/output_{resol}/"
dummy_image_path = "/your/path/to/DL3DV-10K-Benchmark/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/gaussian_splat/"
folder_path_each = os.listdir(data_path)
# folder_path_each.remove('.ipynb_checkpoints')
num_epochs = 200000
save_path = f"/your/path/to/DL3DV-10K/after/3DGS/optimization/gaussian-splatting/train_vae_{resol}/"  # train_vae    train_ae_only2
bch_size = 111 
k_rendering_loss = 1000
enable_rendering_loss = 0
label_gt = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).cuda()
L2=torch.nn.CrossEntropyLoss()
LBCE = torch.nn.BCELoss()
chamferDist = ChamferDistance()
sinkhorn_eff = SamplesLoss(loss="sinkhorn", p=1, blur=.05)
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
    dummy_image_path = "/home/qgao/sensei-fs-link/Dataset/scripts/DL3DV-10K-Benchmark/" + folder_path_each[idx_batch] + "/gaussian_splat/"
    model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", 256, False]
    dataset_for_gs = group_extract(model_params_list, model_params_value)
    gaussians = GaussianModel(dataset_for_gs.sh_degree)
    scene = Scene(dataset_for_gs, gaussians)
    # train_dataset = scene.getTrainCameras()
    viewpoint_stack.append(scene.getTrainCameras().copy())
    training_setup_for_gs = group_extract(optimization_params_list, optimization_params_value)
    pipe = group_extract(pipeline_params_list, pipeline_params_value)


background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")


### fully connected VAE
class GS_encoder(nn.Module):
    def __init__(self, D=8, W=256, input_ch=56, skip=[4], output_ch=4):
        super(GS_encoder, self).__init__()
        self.tau = 0.5
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skip
        self.output_ch = output_ch
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch,W)])
        for i in range(D-1):
            self.pts_linears.append(nn.Linear(W, W))
            self.pts_linears.append(nn.BatchNorm1d(W))
        
        self.output_linear_0 = nn.Linear(W, output_ch)
        self.output_linear_1 = nn.Linear(W, output_ch)

        self.ones_init = torch.ones([1, output_ch])
        self.scale_0 = nn.Parameter(self.ones_init)
        # self.scale_i = nn.Parameter(W)
        #self.scale_1 = nn.Linear(1,output_ch)
        
        #self.class_layer = nn.Linear(output_ch, 2)
        self.act = nn.LeakyReLU(0.2)
       
        self.bn_layer_output = nn.BatchNorm1d(output_ch)
    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = F.relu(x)

        # x = self.bn_layer_inner(x)
        # mu = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.scale_0)) * self.bn_layer_output(self.output_linear_0(x))
        # log_var = torch.sqrt((1 - self.tau) * torch.sigmoid(-self.scale_0)) * self.bn_layer_output(self.output_linear_1(x))
       
        # mu = self.bn_layer_output(self.output_linear_0(x))
        # log_var = self.bn_layer_output(self.output_linear_1(x))
        
        mu = self.output_linear_0(x)
        log_var = self.output_linear_1(x)
        #label = F.sigmoid(self.class_layer(x))
        return mu, log_var #x, label

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
        
        self.output_linear = nn.Linear(W, output_ch)
        self.act = nn.LeakyReLU(0.2)
        self.bn_layer_output = nn.BatchNorm1d(output_ch)
        
    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = F.relu(x)
            # x = self.act(x)
        # x = self.bn_layer_output(self.output_linear(x))
        x = self.output_linear(x)
        return x

# torch.manual_seed(0)
# eps = torch.randn([bch_size, 16384])
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # self.encoder = GS_encoder(8,256,14,[4],14)
        # self.decoder = GS_decoder(8,256,14,[4],14)
        self.encoder = GS_encoder(4,256,14*resol**2,[4],16384)
        self.decoder = GS_decoder(4,256,16384,[4],14*resol**2)
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
    






gs_autoencoder = Network().cuda() 
# gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, str(int(num_epochs)))))
optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=1e-4, betas=[0.9, 0.999])

# eval
# num = 100000
# which_scene = 0
# gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, str(int(num)))))
# gs_autoencoder.eval()
# gs_params_path_each = data_path + folder_path_each[which_scene] + f"/point_cloud/iteration_30000/point_cloud_{resol}.ply"
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
# gs_full_params_norm = ((gs_full_params.reshape([resol, resol, 14]) - norm_min_o) / (norm_max_o - norm_min_o))
# # gs_full_params_norm_pe = torch.zeros_like(gs_full_params_norm).cuda()
# # for pe_ch in range(0,3):
# #     gs_full_params_norm_pe[:,:,pe_ch] = gs_full_params_norm[:,:,pe_ch] - gs_full_params_norm[:,:,pe_ch].mean()
# # gs_full_params_norm_pe[:,:,3:] = gs_full_params_norm_pe[:,:,3:]

# #perm_inputs = gs_full_params_norm[torch.randperm(gs_full_params_norm.size()[0])]
# perm_inputs = gs_full_params_norm
# perm_inputs = perm_inputs.reshape([1, -1])
# # for random row-permutation
# # mu, log_var, z, UV_gs_recover = gs_autoencoder(gs_full_params[torch.randperm(gs_full_params.size()[0])].reshape([1, -1]))

# mix_weight = 0.0
# mix_latent = ((torch.randn_like(latents).to(latents.device)+mu[which_scene])*mix_weight + (torch.randn_like(latents).to(latents.device)+mu[which_scene+2]))*(1-mix_weight)
# UV_gs_recover = gs_autoencoder.decode(mix_latent)[0] #torch.randn_like(latents).to(latents.device)+mu[which_scene])[which_scene]

# UV_gs_recover = gs_autoencoder.decode(torch.randn_like(latents).to(latents.device)+mu[which_scene])[which_scene]


# # UV_gs_recover = gs_full_params.reshape([1, -1])
# # mu, std, z, UV_gs_recover = gs_autoencoder(perm_inputs)
# UV_gs_recover = UV_gs_recover.reshape([resol, resol, 14])# * (norm_max_o-norm_min_o) + norm_min_o
# recovered_idx = UV_gs_recover.reshape(-1,14)
# gaussians._xyz = recovered_idx[:,:3]
# gaussians._features_dc = recovered_idx[:,3:6][:,None,:]
# gaussians._features_rest = torch.zeros([recovered_idx.shape[0], 0, 3]).to(recovered_idx.device)
# gaussians._opacity = recovered_idx[:, 6][:,None]
# gaussians._scaling = recovered_idx[:, 7:10]
# gaussians._rotation = recovered_idx[:, 10:14]
# transform = T.ToPILImage()
# view_1 = viewpoint_stack[which_scene][1]
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
folder_path_each = folder_path_each[:bch_size]
for i in range(len(folder_path_each)):
    gs_params_path_each = data_path + folder_path_each[i] + f"/point_cloud/iteration_30000/point_cloud_{resol}.ply"
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

    
    # gs_full_params_norm = np.concatenate((xyz_norm, color_rgb_norm, opacity_norm[:,None], scale_norm, rot_norm), axis=1)
    gs_full_params = np.concatenate((xyz, color_rgb, opacity[:,None], scale, rot), axis=1)
    UV_gs.append(gs_full_params.reshape([resol, resol, 14]))
    
    gs_norm_factor = np.concatenate((xyz_norm_fact, rgb_norm_fact, opacity_norm_fact, scale_norm_fact, rot_norm_fact), axis=1)
    UV_gs_norm_factors.append(gs_norm_factor)
    
UV_gs = torch.tensor(np.stack(UV_gs, axis=0)).cuda() #.reshape([len(folder_path_each), 64, 64, 56])
#UV_gs_xyz_norm = torch.tensor(np.stack(UV_gs_xyz_norm, axis=0)).cuda()
UV_gs_norm_factors = torch.tensor(np.stack(UV_gs_norm_factors, axis=0)).cuda()
norm_max = torch.zeros([UV_gs.shape[1], UV_gs.shape[2], UV_gs.shape[3]]).cuda()
norm_min = torch.zeros([UV_gs.shape[1], UV_gs.shape[2], UV_gs.shape[3]]).cuda()

# for ch in range(UV_gs_batch.shape[-1]):
#     norm_max[:,:,ch] = UV_gs_batch[:,:,:,ch].max()
#     norm_min[:,:,ch] = UV_gs_batch[:,:,:,ch].min() 
#     UV_gs_batch[:,:,:,ch] = (UV_gs_batch[:,:,:,ch] - norm_min[:,:,ch]) / (norm_max[:,:,ch] - norm_min[:,:,ch])
    
for epoch in tqdm(range(num_epochs)):
       if epoch % 2 == 0 and random_permute == 1:
           UV_gs = UV_gs[:,torch.randperm(UV_gs.size()[1])]
       for bch in range(0, len(folder_path_each), bch_size):
           UV_gs_batch = UV_gs[0:bch_size]
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
           mu, log_var, z, UV_gs_recover = gs_autoencoder(UV_gs_batch.reshape([bch_size, -1]))
           UV_gs_recover = UV_gs_recover.reshape([UV_gs_batch.shape[0], resol, resol, 14]) 
           
           if enable_rendering_loss == 1:
              if epoch % k_rendering_loss == 0: #and epoch >= k_rendering_loss:
                 # for idx_batch in range(len(UV_gs_batch)):
                 viewpoint_stack_test = []
                 random_idx = random.sample(range(0, len(UV_gs_batch)), 2) 
                 for iik in range(len(random_idx)): 
                     idx_batch = random_idx[iik]
                     dummy_image_path = "/your/path/to/DL3DV-10K-Benchmark/" + folder_path_each[idx_batch] +"/gaussian_splat/"
                     model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", resol, False]
                     dataset_for_gs = group_extract(model_params_list, model_params_value)
                     gaussians = GaussianModel(dataset_for_gs.sh_degree)
                     scene = Scene(dataset_for_gs, gaussians)
                     # train_dataset = scene.getTrainCameras()
                     viewpoint_stack_test.append(scene.getTrainCameras().copy())
                     training_setup_for_gs = group_extract(optimization_params_list, optimization_params_value)
                     pipe = group_extract(pipeline_params_list, pipeline_params_value)
                  
                     viewpoint = viewpoint_stack_test[0]   #[idx_batch]
                     recovered_idx = UV_gs_recover[idx_batch].reshape([resol,resol,14])
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
           if loss_usage == "L1":
              loss += torch.norm(UV_gs_recover.reshape([bch_size, -1, 14]) - UV_gs_batch.reshape([bch_size, -1, 14]), p=2)/len(UV_gs_batch) + 0.001*KL_loss + 10*loss_render
           elif loss_usage == "chamfer":
              loss += torch.mean(chamferDist(UV_gs_batch.reshape([bch_size, -1, 14])[:,:,:3], UV_gs_recover.reshape([bch_size, -1, 14])[:,:,:3])) + 0.01*torch.mean(chamferDist(UV_gs_batch.reshape([bch_size, -1, 14])[:,:,3:], UV_gs_recover.reshape([bch_size, -1, 14])[:,:,3:])) + 0.0001*KL_loss + 10*loss_render
           elif loss_usage == "sinkhorn":
                sinkhorn_loss_ = sinkhorn_eff(UV_gs_batch.reshape([bch_size, -1, 14]), UV_gs_recover.reshape([bch_size, -1, 14])).mean()
                loss += sinkhorn_loss_ + 0.001*KL_loss + 10*loss_render
        
           if epoch % 100 == 0: 
              print(f"loss={loss.item()}  ,  kl_loss = {KL_loss.item()}")
           if epoch % 1000 == 0:
                gs_autoencoder.eval()
               # test the reconstruction quality
                recovered_1 = UV_gs_recover[0].reshape([resol,resol,14]) 
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
                if epoch >= 100000 and epoch % 10000 == 0:
                    torch.save(mu, f"{save_path}gs_mu_{epoch}.pt")
                    torch.save(log_var, f"{save_path}gs_var_{epoch}.pt")
                    torch.save(z, f"{save_path}gs_emb_{epoch}.pt")
                    torch.save(norm_min, f"{save_path}norm_min_{epoch}.pt")
                    torch.save(norm_max, f"{save_path}norm_max_{epoch}.pt")
                    
                    torch.save(UV_gs_norm_factors, f"{save_path}norm_xyz_{epoch}.pt")
                    torch.save(gs_autoencoder.state_dict(), os.path.join(save_path, str(int(epoch))))
                gs_autoencoder.train()
           loss.backward()
           optimizer.step()




# test the reconstruction quality
recovered_1 = UV_gs_recover[0].reshape([resol,resol,14])
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

torch.save(norm_min, f"{save_path}norm_min_{epoch+1}.pt")
torch.save(norm_max, f"{save_path}norm_max_{epoch+1}.pt")
torch.save(UV_gs_norm_factors, f"{save_path}norm_xyz_{epoch+1}.pt")
torch.save(gs_autoencoder.state_dict(), os.path.join(save_path, str(int(epoch+1))))

        
        
        
    
    
    
    
    