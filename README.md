> This repository contains a modified version of the official implementation [`Splatter Image: Ultra-Fast Single-View 3D Reconstruction' CVPR 2024 model](https://github.com/szymanowiczs/splatter-image) to include learning from depth maps, undertaken as a project for our degree studies.
> The original model is an ultra-fast approach to 3D reconstruction from a single 2D image using 3D Gaussian Splattings. 
> This README details the original concept, our hypothesis, the modifications made, our results and instructions on how to run the code.


<!-- TOC -->
## Table Of Contents
- [Enhancing 3D Gaussian Splatting using Depth Maps](#enhancing-3d-gaussian-splatting-using-depth-maps)
  - [What is 3D Gaussian Splatting?](#what-is-3d-gaussian-splatting)
  - [What we did](#what-we-did)
- [Results](#results)
   - [Losses and visualization](#losses-and-visualization)
   - [Conclusions](#conclusions)
- [Installation and Training](#installation-and-training)
   - [Installation](#installation)
   - [Extracting depth maps](#extracting-depth-maps)
   - [Training the model](#training-the-model)

# Enhancing 3D Gaussian Splatting using Depth Maps

## What is 3D Gaussian Splatting? 
3D Gaussian Splatting is a method that reconstructs 3D geometry from a single image by rendering Gaussian splats in 3D space.  
The technique achieves ultra-fast rendering speeds and allows for generating novel 3D views from a single 2D input image.  
For more information, visit the original [GitHub repository](https://github.com/szymanowiczs/splatter-image) or refer to the [paper](https://arxiv.org/pdf/2312.13150).

## What we did

 Our project aimed to investigate whether augmenting the model with additional input information—in this case, depth maps—could improve convergence speed, accuracy, or overall efficiency.
To test this hypothesis, we carried out the following steps:

1. Depth Map Extraction:
Using the original pre-trained model, we extracted and saved depth maps for all the images in our training, testing, and validation datasets.
This was done to retrieve depth maps that we reliabley know work with the model.

2. Model Architecture Modification:
The U-Net architecture was updated to accept an additional input channel for the depth maps, allowing the model to leverage depth information during training.

3. Model Evaluation:
Both the original and modified models were trained and evaluated on the same datasets and starting weights to measure the impact of including depth maps alongside images.

4. Depth Visualization:
During training, we extracted and visualized the depth map outputs of both models. This provided a direct comparison of depth reconstruction progress and results.

## Dataset, and experiment setup
For our experiment, we exclusively used the [srn_cars dataset](https://drive.google.com/file/d/19yDsEJjx9zNpOKz9o6AaK-E8ED6taJWU/view?usp=drive_link), as it provided a sufficiently large and diverse collection of samples while being manageable within the constraints of our hardware.
The modified model was trained on 70% of the dataset, which took approximately 8 hours including both stages.  
While the original authors trained their model for a significantly longer duration and on stronger hardware, we believe the chosen training time was sufficient to validate our hypothesis as per our results.  
Further training could likely enhance the results but was beyond the scope of this experiment.

For our setup, we used:

GPU: NVIDIA Geforce RTX 3080 with 10GB VRAM.

CPU: Intel i7-11700KF.

RAM: 16GB.

Software: Python 3.12.4 and CUDA 12.4. For other libraries please refer to the requirements.txt file.


# Results
## Losses and visualization:
We followed the original model's method of having two parts for the training, the first without LPIPS, followed by fine-tuning with LPIPS.

#### Training Loss

![image](https://i.imgur.com/fOHXsrc.png) ![image](https://i.imgur.com/t7TntPA.png)
![image](https://i.imgur.com/pyrjg6e.png) ![image](https://i.imgur.com/vShNi8L.png)

* Base Case: The model trained without depth images.
* Pretrained Depth: The model trained with depth images extracted using the original trained model.

#### Additional Losses
Further comparisons were made using additional loss metrics to quantify the impact of depth images:
![images](https://i.imgur.com/7rtMl5s.png)

#### Renders
Visual comparisons between the rendered outputs and the ground truth:

(renders: left - render by the model, right - ground truth)

![images](https://i.imgur.com/saRDDUo.gif) ![images](https://i.imgur.com/rhM9fKv.gif)
![images](https://i.imgur.com/GC8o1RF.gif) ![images](https://i.imgur.com/2O9Kdbc.gif)


## Conclusions

From our evaluation, we observed a significant improvement in the model's learning by incorporating depth images, as evidenced by both visual results and loss metrics.

To derive stronger conclusions, extended training durations and the exploration of alternative depth generation methods—such as MiDaS, DPT, or MegaDepth—could provide valuable insights and further enhance the model's performance. Furthermore, these findings establish a potentially promising foundation for leveraging depth maps to significantly enhance the learning capabilities of neural networks.


# Installation and Training

## Installation

1. Create a conda environment: 
```
conda create --name splatter-image
conda activate splatter-image
```

Install Pytorch following [official instructions](https://pytorch.org). Pytorch / Python / Pytorch3D combination that was verified to work is:
- Python 3.8, Pytorch 1.13.0, CUDA 11.6, Pytorch3D 0.7.2
Alternatively, you can create a separate environment with Pytorch3D 0.7.2, which you use just for CO3D data preprocessing. Then, once CO3D had been preprocessed, you can use these combinations of Python / Pytorch too. 
- Python 3.7, Pytorch 1.12.1, CUDA 11.6
- Python 3.8, Pytorch 2.1.1, CUDA 12.1
- Python 3.12.4, Pytorch 2.4.0, CUDA 12.4 (Ours)

Install other requirements:
```
pip install -r requirements.txt
```

2. Install Gaussian Splatting renderer, i.e. the library for rendering a Gaussian Point cloud to an image. To do so, pull the [Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting/tree/main) and, with your conda environment activated, run `pip install submodules/diff-gaussian-rasterization`. You will need to meet the [hardware and software requirements](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md#hardware-requirements).

3. Download the srn_cars.zip file from the [PixelNeRF data folder](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR?usp=sharing).

4. Unzip the data file and change `SHAPENET_DATASET_ROOT` in `datasets/srn.py` to the parent folder of the unzipped folder. For example, if your folder structure is: `/home/user/SRN/srn_cars/cars_train`, in `datasets/srn.py` set  `SHAPENET_DATASET_ROOT="/home/user/SRN"`.


## Extracting depth maps
Before extracting the depth maps, make sure to switch to the **saving_depth_images branch**. Once on the correct branch, use the following commands to extract depth maps for each dataset split (train, validation, and test):
```
python generate_depth.py cars --split train
```
```
python generate_depth.py cars --split val
```
```
python generate_depth.py cars --split test
```
The depth images will be saved under the folder "depth_images". The folder path can be changed in the 'generate_depth.py' file.


## Training the model
Before continuing, make sure to **switch to the main branch.**

The model is trained in two stages, first without LPIPS, followed by fine-tuning with LPIPS.
1. The first stage is ran with:
      ```
      python train_network.py +dataset=cars
      ```
      Once it is completed, place the output directory path in configs/experiment/lpips_$experiment_name.yaml in the option `opt.pretrained_ckpt` (by default set to null).
2. Run second stage with:
      ```
      python train_network.py +dataset=cars +experiment=lpips_100k.yaml
      ```
      Remember to place the directory of the model from the first stage in the appropriate .yaml file before launching the second stage.



##

### Thank you for reading our repository!
