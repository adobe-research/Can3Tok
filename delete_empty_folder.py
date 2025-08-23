# sometimes running 3DGS on DL3DV-10K dataset has some empty folders or incomplete folders due to unexpected errors, 
# this script is to delete those empty or incomplete folders from child folders to parent folders with less pain if you don't
# want to delete them manually.

import os
import shutil

path = f"/your/DL3DV-10K/path/"  
folders = os.listdir(path) 
i=0
num_folders = len(folders)


for folder in folders:
    i += 1
    if os.path.isfile(path+folder+"/input.ply"):
        os.remove(path+folder+"/input.ply")
    query_path = path+folder+"/point_cloud"  
    if os.path.isdir(query_path+"/iteration_7000"):
        shutil.rmtree(query_path+"/iteration_7000")
    if not os.path.isdir(query_path) or not os.path.isdir(query_path+"/iteration_30000") or not os.path.isfile(query_path+'/iteration_30000/point_cloud_316.ply'):  # if folder
        shutil.rmtree(path+folder)  # delete empty folder

    print(f"{i}/{num_folders} complete!!!")

