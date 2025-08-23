# This is the script for downsampling the 3D Gaussians in the DL3DV-10K dataset.
# It is not needed if you want to use all the SfM points as initialization for 3DGS, but it is useful if you want to downsample the SfM points to a fixed number of Gaussians per scene.

from plyfile import PlyData, PlyElement
import numpy as np
import os
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

# change this data_path to your data path
data_path = "/your/path/to/DL3DV-10K/"

folder_path_each = os.listdir(data_path)

# delete some files automatically generated during dataset downloading
folder_path_each.remove('benchmark-meta.csv')
folder_path_each.remove('.cache')
folder_path_each.remove('.huggingface')
# sub_folder_path_each_1 = folder_path_each[0:70]
# number of per-scene SfM points to use as initialization for 3DGS, e.g. 128*128 = 16384, 200*200 = 40000, 512*512 = 262144
reso_Gaussian = 128


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

for i in range(len(folder_path_each)):
    path = data_path + folder_path_each[i] + "/gaussian_splat/sparse/0/points3D.ply"
   
    if os.path.isfile(path): 
       plydata = PlyData.read(path)
    else:
       print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
       bin_path = data_path + sub_folder_path_each_1[i] + "/gaussian_splat/sparse/0/points3D.bin"
       xyz, rgb, _ = read_points3D_binary(bin_path)
       storePly(path, xyz, rgb)
       plydata = PlyData.read(path)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        return l
    
    # read xyz from ply (output by colmap)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    
    # read nx_ny_nz from ply (output by colmap)
    normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
                       np.asarray(plydata.elements[0]["ny"]),
                       np.asarray(plydata.elements[0]["nz"])),  axis=1)
    
    # read RGB from ply (output by colmap)
    color_rgb = np.stack((np.asarray(plydata.elements[0]["red"]),
                          np.asarray(plydata.elements[0]["green"]),
                          np.asarray(plydata.elements[0]["blue"])),  axis=1)
    
    scales = np.ones_like(xyz)
    
    rots = np.ones([xyz.shape[0],4])
    
    opacity = np.ones([xyz.shape[0]])
    
    # random index
    idx = np.random.randint(0, len(plydata.elements[0]["x"]), size = reso_Gaussian**2) # 16384 for 128*128; 262144 for 512*512
    
    # randomly downsample reso_Gaussian**2 (128*128) samples from initial sfm points
    xyz_ds = xyz[idx, :]  # 128*128 = 16384 
    rgb_ds = color_rgb[idx, :]
    normals_ds = normals[idx, :] 
    scales_ds = scales[idx, :]
    rots_ds = rots[idx, :]
    opacity_ds = opacity[idx]


    #### save the downsample initial sfm points as .ply files
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
    dtype_full = [(attribute, 'f4') for attribute in l]
    elmts = np.empty(xyz_ds.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz_ds, normals_ds, rgb_ds), axis=1)
    elmts[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elmts, 'vertex')
    PlyData([el]).write(data_path + sub_folder_path_each_1[i] + f"/gaussian_splat/sparse/0/points3D_{reso_Gaussian}.ply")

    # run Gaussian Splatting on this scene
    bash_script = f'python train.py -s {data_path}{sub_folder_path_each_1[i]}/gaussian_splat/'
    print(bash_script)
    os.system(bash_script)
    print(f"{i}-th scene finishied!!!!")
    