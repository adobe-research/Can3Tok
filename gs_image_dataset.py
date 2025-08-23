import os
import torch
from PIL import Image
from torch.utils import data
import numpy as np
from torch.utils.data import DataLoader
from voxelize import voxelize
from plyfile import PlyData, PlyElement
import cv2
import torchvision.transforms as transforms

os.chdir("/mnt/localssd/images/")
class gs_image_dataset(data.Dataset):
    def __init__(self, root, resol, random_permute = False, train=True):
        self.data_path = root
        self.resol = resol
        self.random_permute = random_permute
        self.folder_path_each = os.listdir(self.data_path)[:900]  # 900/985
        self.z_path = np.load("/home/qgao/sensei-fs-link/gaussian-splatting/dl3dv_test/image2latent.npy")
        self.transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to PyTorch tensor, scales image to [0, 1]
        transforms.Resize((540, 960),antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example mean/std for normalization
        ])

    def __getitem__(self, index):
        image_path_dir = os.listdir(self.data_path + self.folder_path_each[index]) 
        img_stack = []
        for i in range(len(image_path_dir)):
            img_pre = cv2.imread(self.data_path + self.folder_path_each[index]+'/'+image_path_dir[i])
            img_post = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_post)
            img_stack.append(self.transform(pil_image))
        # image_pick = img_stack[np.random.choice(len(img_stack), 1).item()]
        image_pick = img_stack[0]
    
            
        z = self.z_path[index][:16384]
        
        # gs_params_path_each = self.data_path + self.folder_path_each[index] + f"/point_cloud/iteration_30000/gs_filtered.ply"
        # plydata = PlyData.read(gs_params_path_each)
        # xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
        #                 np.asarray(plydata.elements[0]["y"]),
        #                 np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        # color_rgb = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
        #                       np.asarray(plydata.elements[0]["f_dc_1"]),
        #                       np.asarray(plydata.elements[0]["f_dc_2"])),  axis=1)
        
        # opacity = np.asarray(plydata.elements[0]["opacity"])
        
        # scale = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
        #                   np.asarray(plydata.elements[0]["scale_1"]),
        #                   np.asarray(plydata.elements[0]["scale_2"])),  axis=1)
        
        # rot = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
        #                 np.asarray(plydata.elements[0]["rot_1"]),
        #                 np.asarray(plydata.elements[0]["rot_2"]),
        #                 np.asarray(plydata.elements[0]["rot_3"])),  axis=1)
        
     
        # return {
        #         'z': z,
        #         'image': image_pick,
        #         }
        return z, image_pick

    def __len__(self):
        return len(self.folder_path_each)