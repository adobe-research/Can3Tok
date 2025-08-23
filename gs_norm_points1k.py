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
import json
from utils.sh_utils import RGB2SH, SH2RGB
# import open3d as o3d

resol = 200

# the path of camera information of the DL3DV dataset
cameras_read_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/output_200/"
# cameras_read_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/dl3dv-1k/"
# cameras_read_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/dl3dv_test/"
# cameras_read_path = f"/mnt/localssd/dl3dv-1k/"

camera_folder_path_each = os.listdir(cameras_read_path)
# camera_folder_path_each.remove('benchmark-meta.csv')
# camera_folder_path_each.remove('.cache')
# camera_folder_path_each.remove('.ipynb_checkpoints')
# camera_folder_path_each.remove('.huggingface')

# the path of the 3DGS reconstruction results without normalization
gs_output_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/output_200/"
# gs_output_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/dl3dv-1k/"
# gs_output_path = f"/home/qgao/sensei-fs-link/gaussian-splatting/dl3dv_test/"
# gs_output_path = f"/mnt/localssd/dl3dv-1k/"
gs_folder_path_each = os.listdir(gs_output_path)
gs_folder_path_each.remove('.ipynb_checkpoints')


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
    radius = diagonal * 1.1  # 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale



# def normalize_cam_dict(T):
    
#     translate, scale = get_tf_cams(T, target_radius=1.0)

    # if in_geometry_file is not None and out_geometry_file is not None:
    #     # check this page if you encounter issue in file io: http://www.open3d.org/docs/0.9.0/tutorial/Basic/file_io.html
    #     geometry = o3d.io.read_triangle_mesh(in_geometry_file)
        
    #     tf_translate = np.eye(4)
    #     tf_translate[:3, 3:4] = translate
    #     tf_scale = np.eye(4)
    #     tf_scale[:3, :3] *= scale
    #     tf = np.matmul(tf_scale, tf_translate)

    #     geometry_norm = geometry.transform(tf)
    #     o3d.io.write_triangle_mesh(out_geometry_file, geometry_norm)
  
    # def transform_pose(W2C, translate, scale):
    #     C2W = np.linalg.inv(W2C)
    #     cam_center = C2W[:3, 3]
    #     cam_center = (cam_center + translate) * scale
    #     C2W[:3, 3] = cam_center
    #     return np.linalg.inv(C2W)

    # out_cam_dict = copy.deepcopy(in_cam_dict)
    # for img_name in out_cam_dict:
    #     W2C = np.array(out_cam_dict[img_name]['W2C']).reshape((4, 4))
    #     W2C = transform_pose(W2C, translate, scale)
    #     assert(np.isclose(np.linalg.det(W2C[:3, :3]), 1.))
    #     out_cam_dict[img_name]['W2C'] = list(W2C.flatten())

    #with open(out_cam_dict_file, 'w') as fp:
    #    json.dump(out_cam_dict, fp, indent=2, sort_keys=True)

    


UV_gs_norm = []

for i in range(len(gs_folder_path_each)):
    gs_params_path_each = gs_output_path + gs_folder_path_each[i] + f"/point_cloud/iteration_30000/point_cloud_{resol}.ply"
    # gs_params_path_each = f"output_128/mipnerf360_stump/point_cloud/iteration_30000/point_cloud_128.ply"

    
    # UV_gs_reshape = gaussians.load_ply(gs_params_path_each)
    plydata = PlyData.read(gs_params_path_each)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    
    # normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
    #                np.asarray(plydata.elements[0]["ny"]),
    #                np.asarray(plydata.elements[0]["nz"])),  axis=1)
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

    

    # path = cameras_read_path + gs_folder_path_each[i] + f"/gaussian_splat/"
    # cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    # cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    # cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    # cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    # T, K_intrinsic = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics)
    translate, scale_factor = get_tf_cams(xyz, target_radius=7.0)

    ### normalization term with respect to sfm_point
    # sfm_point_trans = -np.mean(xyz, axis=0, keepdims=True)
    # sfm_point_dist = np.linalg.norm(xyz + sfm_point_trans, axis=1, keepdims=True)
    # sfm_point_scale = 1/(np.max(sfm_point_dist)*1.1)  #1.1
    norm_factor = np.concatenate([translate.squeeze(), np.array([scale_factor])])

 
    ##### transformation matrix
    # tf_translate = np.eye(4)
    # tf_translate[:3, 3:4] = translate
    # tf_scale = np.eye(4)
    # tf_scale[:3, :3] *= scale
    # tf = np.matmul(tf_scale, tf_translate)
    
    xyz_norm = (xyz + translate)*scale_factor
    scale_norm = scale + np.log(scale_factor)
    # if gs_folder_path_each[i] == "2dc9d40d1b4f0545a7213a20d00f7dd033346743b935c87bbb8a69ba8a4a634d":
    #    import pdb;pdb.set_trace()
    #    a=1
    # scale_perspective = np.ones((scale.shape[0],4))
    # scale_perspective[:,:3] = scale*scale_factor
    # scale_n = np.matmul(scale_perspective, K_intrinsic.transpose(0,1).numpy())
    # scale_norm = (scale_n/(scale_n[:,3][:,None].repeat(4,axis=1)))[:,:3]    

    gs_full_params = np.concatenate((xyz_norm, color_rgb, opacity[:,None], scale_norm, rot), axis=1)

   
    # the path you specifying for storing the normalized 3DGS reconstruction results
    gs_norm_path_write = gs_output_path + gs_folder_path_each[i] + f"/point_cloud/iteration_30000/point_cloud_{resol}_norm.ply"
    # gs_norm_path_write = f"output_128/mipnerf360_stump/point_cloud/iteration_30000/point_cloud_128_norm.ply"
    
    mkdir_p(os.path.dirname(gs_norm_path_write))
    normals = np.zeros_like(xyz_norm)
    f_dc = torch.tensor(gs_full_params[:,3:6][:,None,:]).transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = torch.zeros([gs_full_params.shape[0], 0, 3]).transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gs_full_params[:, 6][:,None]
    scale = gs_full_params[:, 7:10]
    rotation = gs_full_params[:, 10:14]
  

    
    l_list = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC

    for pp in range(3):
        l_list.append('f_dc_{}'.format(pp))
    for pp in range(0):
        l_list.append('f_rest_{}'.format(pp))
    l_list.append('opacity')
    for pp in range(scale.shape[1]):
        l_list.append('scale_{}'.format(pp))
    for pp in range(rotation.shape[1]):
        l_list.append('rot_{}'.format(pp))

        
    dtype_full = [(attribute, 'f4') for attribute in l_list]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz_norm, normals, f_dc, f_rest, opacities, scale_norm, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
   
    PlyData([el]).write(gs_norm_path_write)

    # factor_path = cameras_read_path + gs_folder_path_each[i] + f"/gaussian_splat/sparse/0/norm_factor.npy"
    # np.save(factor_path,np.stack(norm_factor))
    print(f"{i}/{len(gs_folder_path_each)} normalization finished!!!")

    
    
    
   
 
  

    
    
    

