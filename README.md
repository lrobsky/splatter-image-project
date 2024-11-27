# Splatter Image: Ultra-Fast Single-View 3D Reconstruction (modified experiment)
This repository contains some modification of the "Splatter Image: Ultra-Fast Single-View 3D Reconstruction" model.  
The original model is an ultra-fast approach to 3D reconstruction from a single 2D image using 3D Gaussian Splattings.  
This README details the original concept, the modifications made, and instructions on how to run the code.

## What is 3D gaussian splattings ? 
3D Gaussian Splattings is a method that reconstructs 3D geometry from a single image by rendering Gaussian splats in 3D space.  
The technique achieves ultra-fast rendering speeds and allows for generating novel 3D views from a single 2D input image.  
For more information, visit the original [GitHub repository](https://github.com/szymanowiczs/splatter-image) or refer to the [paper](https://arxiv.org/pdf/2312.13150).

## Dataset, and experiment setup
For our experiment, we exclusively used the Cars dataset, as it provided a sufficiently large and diverse collection of samples while being manageable within the constraints of our hardware.
The modified model was trained on 70% (**?**) of the dataset, which took approximately X(**?**) hours.  
While the original authors trained their model for a significantly longer duration, we believe the chosen training time was sufficient to validate our hypothesis.  
Further training could likely enhance the results but was outside the scope of this experiment.



## What we did differently
The objective of this experiment was to check whether providing the model with additional input information (in our case, depth maps) could make the model convergence faster, improve its accuracy, or have greater overall efficiency.  
To test this hypothesis, we followed these steps:

1. Using the original model, we extracted & saved depth maps for all the images in our train,test and validation data sets.  

2. We updated the U-Net architecture to accept an additional input channel corresponding to the saved depth maps. This modification enabled the model to utilize depth information during training.  

3. Both the original and modified models were trained and evaluated on the same datasets, in order to assess the impact of adding depth maps as an additional input along with the images.    

4. Additionally, we also extracted and visualized the depth map output of both models during training. This allowed us to directly compare the depth reconstruction progress and results between the original and modified models.


## Results :

extra Place for images and what not


# Installation

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

Install other requirements:
```
pip install -r requirements.txt
```

2. Install Gaussian Splatting renderer, i.e. the library for rendering a Gaussian Point cloud to an image. To do so, pull the [Gaussian Splatting repository](https://github.com/graphdeco-inria/gaussian-splatting/tree/main) and, with your conda environment activated, run `pip install submodules/diff-gaussian-rasterization`. You will need to meet the [hardware and software requirements](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md#hardware-requirements).

3. Download the srn_cars.zip file from the [PixelNeRF data folder](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR?usp=sharing).

4. Unzip the data file and change `SHAPENET_DATASET_ROOT` in `datasets/srn.py` to the parent folder of the unzipped folder. For example, if your folder structure is: `/home/user/SRN/srn_cars/cars_train`, in `datasets/srn.py` set  `SHAPENET_DATASET_ROOT="/home/user/SRN"`.

   
# Using this repository

## Extracting depth maps
Before extracting the depth maps, make sure to switch to the **saving_test_images branch**(placeholder?). Once on the correct branch, use the following commands to extract depth maps for each dataset split (train, validation, and test):
```
python eval.py cars --split train
```

```
python eval.py cars --split val/vis 
```
(**not sure which one actually works**)

```
python eval.py cars --split test
```


## Training the model
Before continuing, make sure to **switch back to the main branch!**

The model is trained in two stages, first without LPIPS , followed by fine-tuning with LPIPS.
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




# Thank you for reading our repository!
