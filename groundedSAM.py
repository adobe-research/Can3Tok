import sys
sys.path.append('..')
import os
import json
import numpy as np
import torch
from scene.cameras import Camera
# import torchvision.transforms as T
from plyfile import PlyData, PlyElement
# from scene import Scene, GaussianModel
# from gaussian_renderer import render, network_gui
from scene.dataset_readers import getNerfppNorm
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov,fov2focal
from typing import NamedTuple
from scene.gaussian_model import BasicPointCloud
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from lang_sam import LangSAM
from PIL import Image
import cv2
import pynanoflann
import random
from tqdm import tqdm

data_path = f"/your/path/to/DL3DV-10K/" 
camera_path = f"your/path/to/DL3DV-10K/estimated_camera_parameters/" # only if you save SfM results to /estimated_camera_parameters folder seperately
dummy_image_path = "/your/path/to/DL3DV-10K-Benchmark/07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b/gaussian_splat/"
folder_path_each = os.listdir(data_path)

WARNED = False
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
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
        T = np.array(extr.tvec)
        
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        # image_path = os.path.join(images_folder, os.path.basename(extr.name))
        # image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)

        dummy_image = torch.zeros(3, width, height)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=dummy_image,
                              image_path='', image_name=f'frame_{extr.camera_id:4d}', width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapSceneInfo(path, images, eval, llffhold=8):
    cameras_extrinsic_file = os.path.join(path, "images.bin")
    cameras_intrinsic_file = os.path.join(path, "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path='')
    return scene_info

def loadCam(args_resolution, id, cam_info, resolution_scale):
    
    orig_h = cam_info.image.shape[1]
    orig_w = cam_info.image.shape[2]
    if args_resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args_resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args_resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    resized_image_rgb = torch.zeros(3, resolution[0], resolution[1])
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    device = 'cpu'
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, scale_factor=2.0, data_device=device)

def cameraList_from_camInfos(cam_infos, resolution_scale):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(-1, id, c, resolution_scale))

    return camera_list



class GroupParams:
    pass

def group_extract(param_list, param_value):
    group = GroupParams()
    for idx in range(len(param_list)):
        setattr(group, param_list[idx], param_value[idx])
    return group

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

for i in tqdm(range(len(folder_path_each))): 
    ######## loading from Json file, which sucks due to w2c and c2w typos from 3DGS
    # with open(os.path.join(data_path+folder_path_each[i], "cameras.json"),'rb') as file:
    #    parsed_json = []
    #    try:
    #        parsed_json = json.load(file)
    #    except OSError as exc:
    #        print(exc)
    #    if len(parsed_json) == 0:
    #        break
    #    else:
    #        rotation = np.array(parsed_json[100]["rotation"]) # #100 is an intermediate frame
    #        translation = np.array(parsed_json[100]["position"])
    #        fx = np.array(parsed_json[100]["fx"])
    #        fy = np.array(parsed_json[100]["fy"])
    #        id = np.array(parsed_json[100]["id"])
    #        width = np.array(parsed_json[100]["width"])
    #        height = np.array(parsed_json[100]["height"])
    #        img_name = parsed_json[100]["img_name"]
           # ## trace back to w2c
           # w2c = np.zeros((4, 4))
           # w2c[:3, 3] = translation
           # w2c[:3, :3] = rotation
           # w2c[3, 3] = 1.0
           # Rt = np.linalg.inv(w2c)
           # R_cor = Rt[:3, :3]
           # T_cor = Rt[:3, 3]
           
    # three scenes for testing
    ### statue scene
    # folder_path_each[i] = "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7"
    ### market scene
    # folder_path_each[i] = "25231e5e062b71d1f9b0463219e63a2383d55f3b2cec95f50e20f044d60ef4f6"
    ### restaurant scene
    # folder_path_each[i] = "c37726ce770ac50a2cf5c0f43022f0268e26da0d777cd8e3a3418c4eed03fd94"
    
  
    
    scene_info = readColmapSceneInfo(camera_path+"camera_"+folder_path_each[i], dummy_image_path, eval=False)
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 4)

    # model_params_list = ["sh_degree", "source_path", "model_path", "images", "resolution", "white_background", "data_device",     
    #                      "num_gs_per_scene_end", "eval"]
    # model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", 256, False]
    # pipeline_params_list = ["convert_SHs_python", "compute_cov3D_python", "debug"]
    # pipeline_params_value = [False, False, False]
    # optimization_params_list = ["iterations", "position_lr_init", "position_lr_final", "position_lr_delay_mult", "position_lr_max_steps",
    #                             "feature_lr", "opacity_lr", "scaling_lr", "rotation_lr", "percent_dense", "lambda_dssim", 
    #                             "densification_interval", "opacity_reset_interval", "densify_from_iter", "densify_until_iter", 
    #                             "densify_grad_threshold", "random_background"]
    # optimization_params_value = [35_000, 0.00016, 0.0000016, 0.01, 30_000, 0.0025, 0.05, 0.005, 0.001, 0.01, 0.2, 100, 3000, 500, 15_000,0.0002, False]
    
    # view_to_render = Camera(colmap_id=parsed_json[100]["id"],R=R_cor, T=T_cor, FoVx=fx, FoVy=fy, image=dummy_image,gt_alpha_mask=None,image_name=img_name, uid=id, scale_factor=4)
    
    # model_params_value = [0, dummy_image_path, "", "images", -1, False, "cuda", 256, False]
    # dataset_for_gs = group_extract(model_params_list, model_params_value)
    # pipe = group_extract(pipeline_params_list, pipeline_params_value)
    # gaussians = GaussianModel(dataset_for_gs.sh_degree)

    gs_params_path_each = os.path.join(data_path+folder_path_each[i]+"/point_cloud/iteration_30000/", "point_cloud_316.ply") # this point_cloud_316.ply is the output of 3DGS optimization, change prefix "_316" to your actual output name.
    save_path = data_path+folder_path_each[i]+"/point_cloud/iteration_30000"
    print(save_path)
    plydata = PlyData.read(gs_params_path_each)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    # gaussians._xyz = torch.tensor(xyz).cuda()
        
    color_rgb = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                          np.asarray(plydata.elements[0]["f_dc_1"]),
                          np.asarray(plydata.elements[0]["f_dc_2"])),  axis=1)
    # gaussians._features_dc = torch.tensor(color_rgb).cuda()[:,None,:]
    max_sh_degree = 0 
    # gaussians._features_rest = torch.zeros([color_rgb.shape[0],0,3]).cuda()
    opacity = np.asarray(plydata.elements[0]["opacity"])
    # gaussians._opacity = torch.tensor(opacity[:,None]).cuda()
    
    scale = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                      np.asarray(plydata.elements[0]["scale_1"]),
                      np.asarray(plydata.elements[0]["scale_2"])),  axis=1)
    # gaussians._scaling = torch.tensor(scale).cuda()
        
    rot = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                    np.asarray(plydata.elements[0]["rot_1"]),
                    np.asarray(plydata.elements[0]["rot_2"]),
                    np.asarray(plydata.elements[0]["rot_3"])),  axis=1)
    # gaussians._rotation = torch.tensor(rot).cuda()
    
    # background = torch.tensor([0,0,0], dtype=torch.float32).cuda()

    # we pick out the middle frame
    camera_id_vis = int(len(train_cameras)/2)
    # render_pkg = render(train_cameras[camera_id_vis], gaussians, pipe, background)
    # transform = T.ToPILImage()
    # rendered_image = render_pkg["render"]
    # rendered_image_to_save = transform(rendered_image)
    # rendered_image_to_save.save(save_path+"/perm_gt.png")
    model = LangSAM()
    image_pil = Image.open(save_path+"/perm_gt.png").convert("RGB")
    
    text_prompt = "the most salient region"

    # masks, boxes, phrases, logits = model.predict(image_pil, text_prompt) # due to the conflicts of different langsam versions, this might not work
    langsam_predict = model.predict(image_pil, text_prompt)
    masks = langsam_predict["masks"]
    boxes = langsam_predict["boxes"]
    # for saving the memory
    del model
    del langsam_predict
    del scene_info
    
    if boxes.shape[0]>1: 
        areas = (boxes[:,2] - boxes[:,0])*(boxes[:,3] - boxes[:,1]) 
        masks = np.array(torch.tensor(masks[np.argmin(areas)]).unsqueeze(0))
    if boxes.shape[0]<1:
        masks = np.ones([1, image_pil.size[1], image_pil.size[0]])
        masks = np.array(masks, dtype='bool') 

   
    ###### visualization testing
    # color = (0, 255, 0)  # BGR color of the bounding box
    # thickness = 2  
    #image = cv2.imread(save_path+"/perm_gt.png")
    ## image = cv2.imread("/home/qgao/sensei-fs-link/gaussian-splatting/dl3dv_test/25231e5e062b71d1f9b0463219e63a2383d55f3b2cec95f50e20f044d60ef4f6/images_4/frame_00122.png")
    # cv2.rectangle(image, (int(boxes[0,0].item()),int(boxes[0,1].item())), (int(boxes[0,2].item()),int(boxes[0,3].item())), color, thickness)
    # cv2.imwrite(save_path+"/bbox.png", image)
    #######


    ##### Gaussian projection from 3D into 2D
    gs_xyz_persp = torch.ones([xyz.shape[0], xyz.shape[1]+1])
    gs_xyz_persp[:,:3] = torch.tensor(xyz) # [x,y,z,1]
    gs_to_pixel = torch.mm(torch.tensor(train_cameras[camera_id_vis].full_proj_transform).T, gs_xyz_persp.T.cuda()).cpu()# K*[R,T]*[x,y,z,1]^T
    xy_proj = (gs_to_pixel/gs_to_pixel[3,:])[:2,:]
   
    xy_proj[0,:] = ((xy_proj[0,:]+1)*image_pil.size[0]-1)*0.5
    xy_proj[1,:] = ((xy_proj[1,:]+1)*image_pil.size[1]-1)*0.5
    xy_proj = xy_proj.T.int()
    xy_proj[xy_proj<0] = 0
    xy_proj[xy_proj[:,0]>=image_pil.size[0]] = 0
    xy_proj[xy_proj[:,1]>=image_pil.size[1]] = 0
    gs_filtered_idx = masks[:,xy_proj[:,1],xy_proj[:,0]].T.squeeze(-1)
    


    scale_radius = np.linalg.norm(xyz-xyz.mean(0), axis=1, keepdims=True).max()
    neighbor_n = pynanoflann.KDTree(n_neighbors=40, metric='L2', radius=0.1*scale_radius)
    neighbor_n.fit(xyz)
    neighbor_dist, neighbor_idx = neighbor_n.kneighbors(xyz)
    queue_k = []
    num_k = 40000 # the number of per-scene 3D Gaussians you want Can3Tok to train on.    *8000/16384 16384/10k

    #### belows are 3 different ways to initialize the first init_sample_num Gaussians
    init_sample_num = 20 # the number of initial Gaussians within the bbox you want to start the kNN search with
    #### 1) initial Gaussians from filtered out Gaussians (random pick)
    # all_idx = np.arange(0,gaussians.get_xyz.shape[0])
    # filtered_idx = all_idx[gs_filtered_idx].tolist()
    # init_gs_samples = random.sample(filtered_idx, init_sample_num)
    #### or 2) initial Gaussians from filtered out Gaussians (mean pick)
    all_idx = np.arange(0,xyz.shape[0])
    xyz_mean = xyz[all_idx[torch.tensor(gs_filtered_idx, dtype=torch.int32)],:].mean(0)
    dist = np.linalg.norm(xyz-xyz_mean, axis=1, keepdims=True)
    init_gs_samples = np.argsort(dist,axis=0)[:init_sample_num,0]
    #### or 3) initial Gaussians from filtered out Gaussians (smallest z/depth)
    # all_idx = np.arange(0,gaussians.get_xyz.shape[0])
    # filtered_idx = all_idx[gs_filtered_idx]
    # init_gs_samples_idx = np.argsort(xyz[all_idx[gs_filtered_idx],2])[:init_sample_num]
    # init_gs_samples = filtered_idx[init_gs_samples_idx].tolist()
    del masks
    del boxes
    del image_pil
    del train_cameras
    torch.cuda.empty_cache()
    
    
    for kk in range(init_sample_num):
        queue_k.append(init_gs_samples[kk])
        
    for ik in range(int(num_k)):
        if len(queue_k) < num_k:
           pick_rdm_idx = np.random.randint(len(queue_k), size=20)
           elmts_to_be_added = neighbor_idx[np.array(queue_k,dtype=np.int64)[pick_rdm_idx],:].flatten()
           msk = np.isin(elmts_to_be_added,np.array(queue_k,dtype=np.int64))
           queue_k.extend(elmts_to_be_added[~msk])
        else:
           break
    top_k_idx = np.array(queue_k, dtype=int)[:num_k]
    
    xyz_norm = xyz[top_k_idx]
    color_rgb = color_rgb[top_k_idx]
    opacity = opacity[top_k_idx]
    scale_norm = scale[top_k_idx]
    rot = rot[top_k_idx]


    ###### GS normalization after filtering and save filtered GS as npy
    translate, scale_factor = get_tf_cams(xyz_norm, target_radius=10.0) # target_radius = any value you want 
    xyz_norm = (xyz_norm + translate)*scale_factor
    scale_norm = scale_norm + np.log(scale_factor)
    gs_full_params = np.concatenate((xyz_norm, color_rgb, opacity[:,None], scale_norm, rot), axis=1)
    np.save(save_path+f"/{folder_path_each[i]}.npy", gs_full_params)

    ###### save filtered GS as ply
    gs_filter_path_write = save_path+"/gs_filtered.ply"
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
    elements = np.empty(xyz_norm.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz_norm, normals, f_dc, f_rest, opacities, scale_norm, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(gs_filter_path_write)

    print(f"{i}/{len(folder_path_each)} Semantic-filtering and normalization finished!!!")