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
from lib.pointops.functions import pointops
# from gs_dataset_original import gs_dataset
from gs_dataset import gs_dataset
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.utils.data as Data
from scipy.stats import special_ortho_group

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_usage = "L1" # L1, sinkhorn, chamfer
random_permute = 0
random_rotation = 1
random_shuffle = 1

resol = 200
data_path = f"/your/Dl3DV-10K/path"

dummy_image_path = "your/DL3DV-10K/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/gaussian_splat/"



folder_path_each = os.listdir(data_path)
# folder_path_each.remove('.ipynb_checkpoints')
num_epochs = 200000
save_path = f"/your/save/path"


bch_size = 10 #200
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



background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = x_v.shape; s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=14, k=14):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        
    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo_0, pxo_1, pxo_2):
        p0 = pxo_0  # (n, 3), (n, c), (b)
        x0 = pxo_1
        o0 = pxo_2
        
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
       
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
    
        x = self.cls(x1)
   
        return x


    


######## pointnet VAE
# class GS_encoder(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=56, skip=[4], output_ch=4):
#         super(GS_encoder, self).__init__()
#         self.tau = 0.5
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.skips = skip
#         self.output_ch = output_ch
#         # self.pts_linears = nn.ModuleList([nn.Linear(input_ch,W)])
#         # for i in range(D-1):
#         #     self.pts_linears.append(nn.Linear(W, W))
#         #     self.pts_linears.append(nn.BatchNorm1d(W))
        

#         # self.ones_init = torch.ones([1, output_ch])
#         # self.scale_0 = nn.Parameter(self.ones_init)
        
#         # self.scale_i = nn.Parameter(W)
#         #self.scale_1 = nn.Linear(1,output_ch)
        
#         #self.class_layer = nn.Linear(output_ch, 2)
#         self.act = nn.LeakyReLU(0.2)
       
#         self.bn_layer_output = nn.BatchNorm1d(output_ch)

#         self.conv_linears = nn.ModuleList([nn.Conv1d(in_channels=input_ch, out_channels=W, kernel_size=1)])
#         for i in range(D-1):
#             self.conv_linears.append(nn.Conv1d(in_channels=W, out_channels=W, kernel_size=1))
#             self.conv_linears.append(nn.BatchNorm1d(W))
            
#         self.conv_linears.append(nn.Conv1d(in_channels=W, out_channels=2*W, kernel_size=1))
#         self.conv_linears.append(nn.BatchNorm1d(2*W))
#         self.conv_linears.append(nn.Conv1d(in_channels=2*W, out_channels=4*W, kernel_size=1))
#         self.conv_linears.append(nn.BatchNorm1d(4*W)) 
         
#         self.output_linear_0 = nn.Linear(4*W, output_ch)
#         self.output_linear_1 = nn.Linear(4*W, output_ch)
    
#     def forward(self, x):
#         for i, l in enumerate(self.conv_linears):
#             x = self.conv_linears[i](x)
#             x = F.relu(x)
        
#         # do max pooling 
#         # import pdb;pdb.set_trace()
#         # import pdb;pdb.set_trace()
#         x = torch.max(x, 2, keepdim=True)[0].squeeze()
#         # import pdb;pdb.set_trace()
        
#         x = x.reshape([x.shape[0],-1])
        
#         #x = x.view(-1, self.output_ch)
#         # get the global embedding
        
        
#         # x = self.bn_layer_inner(x)
#         # mu = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.scale_0)) * self.bn_layer_output(self.output_linear_0(x))
#         # log_var = torch.sqrt((1 - self.tau) * torch.sigmoid(-self.scale_0)) * self.bn_layer_output(self.output_linear_1(x))
       
#         mu = self.bn_layer_output(self.output_linear_0(x))
#         log_var = self.bn_layer_output(self.output_linear_1(x))
        
#         # mu = self.output_linear_0(x)
#         # log_var = self.output_linear_1(x)
        
#         #label = F.sigmoid(self.class_layer(x))
#         return mu, log_var #x, label

# # ####### fully-connected mlp decoder
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
        
#         self.output_linear = nn.Linear(in_features=W, out_features=output_ch)
#         # self.bn_layer_output = nn.BatchNorm1d(output_ch)
        
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
#         self.encoder = GS_encoder(4,256,resol**2,[4],64*64*14)
#         self.decoder = GS_decoder(4,256,64*64*14,[4],14*resol**2)
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





# gs_autoencoder = Network().cuda() 
# gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, str(int(num_epochs)))))

ckpt = 0
if torch.cuda.device_count() > 1:
  gs_autoencoder = nn.DataParallel(PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3]))
else:
  gs_autoencoder = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3])
gs_autoencoder.to(device)

optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=1e-4, betas=[0.9, 0.999])


# eval
# num = 200000
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

# # mix_latent = (torch.randn_like(latents).to(latents.device)+mu[which_scene] + torch.randn_like(latents).to(latents.device)+mu[which_scene+2])/2
# # UV_gs_recover = gs_autoencoder.decode(mix_latent)[which_scene]    #(torch.randn_like(latents).to(latents.device)+mu[which_scene])[which_scene]

# UV_gs_recover = gs_autoencoder.decode(latents.to(latents.device))[which_scene]


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


gs_dataset = gs_dataset(data_path, resol = 128, random_permute = True, train=True)
trainDataLoader = Data.DataLoader(dataset=gs_dataset, batch_size=bch_size, shuffle=False, num_workers=12) 
num = 1750
gs_autoencoder.load_state_dict(torch.load(os.path.join(save_path, f"{str(int(num))}_4k")))
gs_autoencoder.eval()
for epoch in tqdm(range(num_epochs)):
    for i_batch, UV_gs_batch in enumerate(trainDataLoader):
        UV_gs_batch = UV_gs_batch[0].type(torch.float32).to(device)
        if epoch % 1 == 0 and epoch > 1 and random_rotation ==1:
              rand_rot_comp = special_ortho_group.rvs(3)
              # print(rand_rot_comp)
              rand_rot = torch.tensor(np.dot(rand_rot_comp, rand_rot_comp.T), dtype = torch.float32).to(UV_gs_batch.device)
              UV_gs_batch[:,:,4:7] = UV_gs_batch[:,:,4:7]@rand_rot


        
        
        loss = 0.0
        loss_render = 0.0
        optimizer.zero_grad()

        UV_gs_batch2sequence_xyz = []
        UV_gs_batch2sequence_feat = []
        for oo_size in range(UV_gs_batch.shape[0]):
            UV_gs_batch2sequence_xyz.append(UV_gs_batch[oo_size,:,4:7])
            UV_gs_batch2sequence_feat.append(UV_gs_batch[oo_size,:,7:])
            
        UV_gs_batch2sequence_xyz = torch.concatenate(UV_gs_batch2sequence_xyz).contiguous()
        UV_gs_batch2sequence_feat = torch.concatenate(UV_gs_batch2sequence_feat).contiguous()
        UV_gs_batch2sequence_o = torch.tensor([UV_gs_batch.shape[1]*(x+1) for x in range(UV_gs_batch.shape[0])], dtype = torch.int32).to(device)



        K = gs_autoencoder(UV_gs_batch2sequence_xyz, UV_gs_batch2sequence_feat, UV_gs_batch2sequence_o)
        UV_gs_recover = K.reshape([UV_gs_batch.shape[0], 40000,14])

        ##### evaluation
        gs_autoencoder.eval()
        K = gs_autoencoder(UV_gs_batch2sequence_xyz, UV_gs_batch2sequence_feat, UV_gs_batch2sequence_o)
        for kp in range(UV_gs_recover.shape[0]):
            recovered_1 = UV_gs_recover[kp].reshape([resol,resol,14]) 
            recovered_1 = recovered_1.reshape(-1,14)
            gaussians._xyz = recovered_1[:,:3]
            gaussians._features_dc = recovered_1[:,3:6][:,None,:]
            gaussians._features_rest = torch.zeros([recovered_1.shape[0], 0, 3]).to(recovered_1.device)
            gaussians._opacity = recovered_1[:, 6][:,None]
            gaussians._scaling = recovered_1[:, 7:10]
            gaussians._rotation = recovered_1[:, 10:14]
            gaussians.save_ply(save_path+f"recovered_{kp}.ply")
            print(f"test_error = {torch.norm(UV_gs_recover.reshape(UV_gs_batch.shape[0],-1,14) - UV_gs_batch[:,:,4:], p=2)/UV_gs_batch.shape[0]}")
    
        if loss_usage == "L1":
              loss += torch.norm(UV_gs_recover - UV_gs_batch[:,:,4:].reshape([UV_gs_recover.shape[0], 40000, 14]), p=2)/UV_gs_batch.shape[0] #+ 0.0001*KL_loss + 10*loss_render
        elif loss_usage == "chamfer":
              loss += torch.mean(chamferDist(UV_gs_batch.reshape([bch_size, -1, 14])[:,:,:3], UV_gs_recover.reshape([bch_size, -1, 14])[:,:,:3])) + 0.01*torch.mean(chamferDist(UV_gs_batch.reshape([bch_size, -1, 14])[:,:,3:], UV_gs_recover.reshape([bch_size, -1, 14])[:,:,3:])) + 0.0001*KL_loss + 10*loss_render
        elif loss_usage == "sinkhorn":
                sinkhorn_loss_ = sinkhorn_eff(UV_gs_batch.contiguous().reshape([bch_size, -1, 14]), UV_gs_recover.contiguous().reshape([bch_size, -1, 14])).mean()
                loss += sinkhorn_loss_ + 0.001*KL_loss + 10*loss_render
     
        if epoch % 10 == 0: 
                print(f"loss={loss.item()}")
        if epoch % 10 == 0:
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
                if epoch >= 50 and epoch % 50 == 0:
                    torch.save(gs_autoencoder.state_dict(), os.path.join(save_path, f"{str(int(epoch))}_4k"))
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

        
        
        
    
    
    
    
    