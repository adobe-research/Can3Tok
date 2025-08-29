<img src="image/USC-Logos.png" width=120px /><img src="./image/Adobe-Logos.png" width=120px />

<div align="center">

# Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians

### ICCV 2025

<p align="center">  
    <a href="https://zerg-overmind.github.io/">Quankai Gao</a><sup>1</sup>,
    <a href="https://iliyan.com/">Iliyan Georgiev</a><sup>2</sup>,
    <a href="https://tuanfeng.github.io/">Tuanfeng Y. Wang</a><sup>2</sup>,
    <a href="https://krsingh.cs.ucdavis.edu/">Krishna Kumar Singh</a><sup>2</sup>,
    <a href="https://viterbi.usc.edu/directory/faculty/Neumann/Ulrich">Ulrich Neumann</a><sup>1+</sup>,
    <a href="https://gorokee.github.io/jsyoon/">Jae Shin Yoon</a><sup>2+</sup>
    <br>
    <sup>1</sup>USC <sup>2</sup>Adobe Research
</p>

</div>

<div align="center">
    <a href="https://zerg-overmind.github.io/Can3Tok.github.io/"><strong>Project Page</strong></a> |
    <a href="https://arxiv.org/abs/2508.01464"><strong>Paper</strong></a> 
</div>

<br>

<div align="center">

</div>


In this project, we introduce Can3Tok, the first 3D scene-level variational autoencoder (VAE) capable of encoding a large number of Gaussian primitives into a low-dimensional latent embedding, which enables high-quality and efficient generative modeling of complex 3D scenes.

## Cloning the Repository
```bash
git clone https://github.com/Zerg-Overmind/Can3Tok.git 
cd Can3Tok
```

## Environment Installation
We provide a conda environment file for easy installation. Please run the following command to create the environment:
```bash 
bash env_in_one_shot.sh
```
and then activate it:
```bash
conda activate can3tok
```
Please refer to the official repo to install [lang-sam](https://github.com/luca-medeiros/lang-segment-anything) for implementing our Semantics-aware filtering as in `groundedSAM.py`. Note that the pytorch version compatible with the latest lang-sam is `torch==2.4.1+cu121` instead of `torch==2.1.0+cu121` in our `env_in_one_shot.sh`, please modify the environment file accordingly if you want to use the latest lang-sam.
##

## Overall Instruction
1. We firstly run structure-from-motion (SfM) on [DL3DV-10K](https://github.com/DL3DV-10K/Dataset) dataset with [COLMAP](https://colmap.github.io/) to get the camera parameters and sparse point clouds i.e. SfM points. 
2. Then, two options are allowed for applying 3DGS optimization on Dl3DV-10K dataset with camera parameters and SfM points initialized as above.
   - Option 1: We first normalize camera parameters (centers/translation only) and SfM points into a unit (or a predefined radius `target_radius` in the code) sphere, and then run 3DGS optimization afterwards. You might want to check `down_sam_init_sfm.py`
   for the details.
   - Option 2: Or, we can run 3DGS optimization first, and then normalize camera parameters (centers/translation only) and the optimized 3D Gaussians into a unit (or a predefined radius `target_radius` in the code) sphere as a post-processing by normalizing their positions and anisotropic scaling factors. 
  Please refer to `sfm_camera_norm.py` for the implementation of normalization. Additionally, please refer to our `train.py` and related scripts for 3DGS optimization, which ensure that the output filenames match the corresponding input scenes from the DL3DV-10K dataset.
3. (optional) We can optionally run Semantics-aware filtering with [lang_sam](https://github.com/luca-medeiros/lang-segment-anything) to filter out the 3D Gaussians that are not relevant to the main objects of interest in the scene.The implementation is provided in `groundedSAM.py` which includes built-in 3DGS normalization—so there is no need to perform normalization (in step 2. above) separately. That is, we can directly run `groundedSAM.py` after running 3DGS optimization (step.1). The output of this step is a filtered 3D Gaussian splatting point cloud, which is saved in the same output folder after 3DGS optimization for each scene. 
4. Finally, we can run Can3Tok training and testing with 3D Gaussians (w/ or w/o filtering) as input. Please refer to `gs_can3tok.py` for the implementation.

## Details of Per-scene 3DGS optimization
To enable uniform training of Can3Tok across thousands of diverse scenes, we enforce a consistent number of 3D Gaussians per scene. A naive approach would be to initialize the 3DGS representation of each scene using the same number of SfM points while disabling densification and pruning. However, this often leads to suboptimal results. Instead, our logic for densification and pruning is as follows:
1. we start densification as official implementation of 3DGS from iteration opt.densify_from_iter, e.g. 7000.
2. we perform densification until iteration opt.densify_until_iter, e.g. 15000.
3. At iteration opt.densify_until_iter, we prune the number of Gaussians to be exactly dataset.num_gs_per_scene_end**2, e.g. 200*200 = 40000.
4. After that, we continue 3DGS optimization until opt.iterations, e.g. 30000.
This is to make sure that we have a fixed number of Gaussians at the end of training for each scene, while with small PSNR degradation. Please refer to the code in `train.py` for details. We also enable the hint and code for starting from a fixed number of SfM points as initialization in `scene/dataset_readers.py`.

If you've already have 3DGS results for DL3DV-10K dataset, you can skip the 3DGS optimization step and directly use `groundedSAM.py` to crop out a user-specific number of Gaussians for each scene, e.g. 40K or 100K etc, for training Can3Tok. You will also need to modify the output size of decoder MLP at [here](https://github.com/adobe-research/Can3Tok/blob/master/model/michelangelo/models/tsal/sal_perceiver.py#L332) to match the input number of 3DGS.


## Training and Evaluation
To train Can3Tok, please run the following command:
```bash
python gs_can3tok.py
```
where you might want to modify the path pointing to the 3D Gaussians path and output path in the script. For evaluation, please uncomment the evaluation part in `gs_can3tok.py`. 
##

## Baselines
We also provide the code for training and evaluating the baselines in `gs_pointtransformer.py`, `gs_ae.py`, `gs_pointvae.py` and etc. Also, please refer to `tsne_exp*` for the t-SNE visualization of the latent space of Can3Tok and baselines.  

## Generative applications
Feel free to explore various generative applications using Can3Tok, such as 3D scene synthesis with various diffusion models!

## Acknowledgement
We would like to thank the authors of the following repositories for their open-source code and datasets, which we built upon in this work:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Lang-Segment-Anything](https://github.com/luca-medeiros/lang-segment-anything)
- [Michelangelo](https://github.com/NeuralCarver/Michelangelo)
- [DL3DV-10K Dataset](https://dl3dv-10k.github.io/DL3DV-10K/)

## Citation
If you find our code or paper useful, please consider citing:
```
@INPROCEEDINGS{gao2023ICCV,
  author = {Quankai Gao and Iliyan Georgiev and Tuanfeng Y. Wang and Krishna Kumar Singh and Ulrich Neumann and Jae Shin Yoon},
  title = {Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2025}
}
```