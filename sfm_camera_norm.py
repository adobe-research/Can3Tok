import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.system_utils import mkdir_p
import torch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

resol = 128 # 128*128 3D Gaussians per scene, 200 or 316...
cameras_read_path = f"/your/path/to/DL3DV-10K/"
camera_folder_path_each = os.listdir(cameras_read_path)
camera_folder_path_each.remove('benchmark-meta.csv')
camera_folder_path_each.remove('.cache')
camera_folder_path_each.remove('.ipynb_checkpoints')
camera_folder_path_each.remove('.huggingface')
gs_output_path = f"/your/output_{resol}/"
gs_folder_path_each = os.listdir(gs_output_path)
# gs_folder_path_each.remove('.ipynb_checkpoints')
def readColmapCameras(cam_extrinsics, cam_intrinsics):
    T = []
    cam_center = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T.append(np.array(extr.tvec))

        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = np.array(extr.tvec)
        Rt[3, 3] = 1.0
        C2W = np.linalg.inv(Rt)
        cam_center.append(C2W[:3, 3])
        
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY)
    sys.stdout.write('\n')
    return np.stack(cam_center), projection_matrix


def get_tf_cams(cam_dict, target_radius=1.):
    # for im_name in cam_dict:
    #     W2C = np.array(cam_dict[im_name]['W2C']).reshape((4, 4))
    #     C2W = np.linalg.inv(W2C)
    #     cam_centers.append(C2W[:3, 3:4])

    def get_center_and_diag(T):
        cam_centers = np.stack(T,axis=0)
        avg_cam_center = np.mean(cam_centers, axis=0, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=1, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_dict)
    radius = diagonal * 1.1 

    translate = -center
    scale = target_radius / radius

    return translate, scale

def fetchPly(path): 
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return positions, colors, normals


norm_factor = []
for i in range(len(camera_folder_path_each)):
    camera_params_folder_path_each = cameras_read_path + camera_folder_path_each[i] + f"/gaussian_splat/sparse/0/images.bin"
    sfm_path_each = cameras_read_path + camera_folder_path_each[i] + f"/gaussian_splat/sparse/0/points3D.ply"

    sfm_norm_path_write = cameras_read_path + camera_folder_path_each[i] + f"/gaussian_splat/sparse/0/points3D_norm_point.ply"
    # UV_gs_reshape = gaussians.load_ply(gs_params_path_each)
    
    # plydata = PlyData.read(sfm_path_each)

    positions, colors, normals = fetchPly(sfm_path_each)

    path = cameras_read_path + camera_folder_path_each[i] + f"/gaussian_splat/"
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    T, K_intrinsic = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics)

    ### normalization term with respect to sfm_point
    target_radius = 10.0
    sfm_point_trans = -np.mean(positions, axis=0, keepdims=True)
    sfm_point_dist = np.linalg.norm(positions + sfm_point_trans, axis=1, keepdims=True)
    sfm_point_scale = target_radius/(np.max(sfm_point_dist) * 1.1) 
    ### norm_factor is to normalize camera translations (while keeping rotation unchanged) w.r.t the normalized sfm points
    norm_factor = np.concatenate([sfm_point_trans.squeeze(), np.array([sfm_point_scale])])
    positions_norm = (positions + sfm_point_trans) * sfm_point_scale

    l_list = ['x', 'y', 'z', 'red', 'green', 'blue', 'nx', 'ny', 'nz']  
    dtype_full = [(attribute, 'f4') for attribute in l_list]
    elements = np.empty(positions_norm.shape[0], dtype=dtype_full)
    attributes = np.concatenate((positions_norm, colors, normals), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(sfm_norm_path_write)

    
    factor_path = cameras_read_path + camera_folder_path_each[i] + f"/gaussian_splat/sparse/0/norm_factor.npy"
    np.save(factor_path,np.stack(norm_factor))
    print(f"{i}/{len(camera_folder_path_each)} normalization finished!!!")



 
  

    
    
    

