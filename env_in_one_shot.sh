conda create -n can3tok python=3.11 -y

conda activate can3tok 

pip install torch==2.1.0+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
# pip install torch==2.4.1+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

pip install git+https://github.com/u1234x1234/pynanoflann.git@0.0.8
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization 
pip install plyfile
pip install tqdm
pip install chamferdist
pip install einops
pip install pytorch_lightning
pip install omegaconf
pip install scikit-image
pip install opencv-python
pip install trimesh
pip install flash-attn --no-build-isolation
pip install pykeops
pip install geomloss
pip install diffusers
pip install transformers
pip install datasets
pip install peft
pip install wandb
pip install spconv-cu120
# pip install sam-2@git+https://github.com/facebookresearch/segment-anything-2@7e1596c0b6462eb1d1ba7e1492430fed95023598
# pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
pip install numpy==1.24.4

