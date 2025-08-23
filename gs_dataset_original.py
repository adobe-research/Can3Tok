import os
import torch
from PIL import Image
from torch.utils import data
import numpy as np
from torch.utils.data import DataLoader
from voxelize import voxelize
from plyfile import PlyData, PlyElement

class gs_dataset(data.Dataset):
    def __init__(self, root, resol, random_permute = False, train=True):
        self.data_path = root
        self.resol = resol
        self.random_permute = random_permute
        self.folder_path_each = os.listdir(self.data_path)[150:1000]

    def __getitem__(self, index):
        gs_params_path_each = self.data_path + self.folder_path_each[index] + f"/point_cloud/iteration_30000/point_cloud_{self.resol}_norm.ply"
        # gs_params_path_each = self.data_path + self.folder_path_each[index] + f"/point_cloud/iteration_30000/gs_filtered.ply"
        plydata = PlyData.read(gs_params_path_each)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        color_rgb = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                              np.asarray(plydata.elements[0]["f_dc_1"]),
                              np.asarray(plydata.elements[0]["f_dc_2"])),  axis=1)
        
        opacity = np.asarray(plydata.elements[0]["opacity"])
        
        scale = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                          np.asarray(plydata.elements[0]["scale_1"]),
                          np.asarray(plydata.elements[0]["scale_2"])),  axis=1)
        
        rot = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                        np.asarray(plydata.elements[0]["rot_1"]),
                        np.asarray(plydata.elements[0]["rot_2"]),
                        np.asarray(plydata.elements[0]["rot_3"])),  axis=1)
        # #### w/o norm
        # random_shift = 30*np.random.random(size=xyz.shape)
        # random_scale = 30*np.random.random(size=xyz.shape)
        # xyz = (xyz + random_shift)*random_scale
        # scale = scale*random_scale
    
        coord_min = np.min(xyz, 0)
        coord = xyz - coord_min
        uniq_idx, count = voxelize(coord, 0.4, 'fnv') # [-8, 8] with voxel_size=0.4    # ravel, fnv
        gs_full_params = np.concatenate((xyz, color_rgb, opacity[:,None], scale, rot), axis=1)  
        ####### centers as PE
        volume_dims = 40
        resolution = 16.0/volume_dims
        origin_offset = np.array([(volume_dims - 1) / 2, (volume_dims - 1) / 2, (volume_dims - 1) / 2]) * resolution
        shifted_points = xyz + origin_offset
        voxel_indices = np.floor(shifted_points / resolution).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, np.array(volume_dims) - 1)
        voxel_centers = (voxel_indices - (np.array(volume_dims) - 1) / 2) * resolution
        
        gs_full_params = np.concatenate((voxel_centers, np.array(uniq_idx)[:,None], gs_full_params), axis=1) 
        
        ##### padding in case...
        # if gs_full_params.shape[0] != 40000:
        #    dummpy_gs_full_params = np.zeros([40000,14],dtype=np.float32)
        #    dummpy_gs_full_params[:gs_full_params.shape[0],:] = gs_full_params
        #    dummpy_gs_full_params[gs_full_params.shape[0],:] = gs_full_params[-1,:]
        #    gs_full_params = dummpy_gs_full_params
            
        # if self.random_permute == True:
        #    gs_full_params = gs_full_params[torch.randperm(gs_full_params.size()[1])]
        # gs_full_params = gs_full_params[uniq_idx]
        return gs_full_params, index

    def __len__(self):
        return len(self.folder_path_each)


