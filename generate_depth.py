import argparse
import json
import os
import sys
import tqdm
from omegaconf import OmegaConf
# sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import lpips as lpips_lib

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from gaussian_renderer import render_predicted
from scene.gaussian_predictor import GaussianSplatPredictor
from datasets.dataset_factory import get_dataset
from utils.loss_utils import ssim as ssim_fn
import shutil


def check_dataset(dataset):
    print(f"Dataset length: {len(dataset)}")
    
    # Print information about a sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    for key, value in sample.items():
        print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")

def check_dataloader(dataloader):
    for batch_idx, data in enumerate(dataloader):
        print(f"Batch index: {batch_idx}")
        print(f"Data keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")
        # Break after first batch for inspection
        break

@torch.no_grad()
def generate_depth_images(model, dataloader, device, model_cfg, save_vis=0, out_folder=None,split='test'):


    if save_vis > 0:

        os.makedirs(out_folder, exist_ok=True)

    bg_color = [1, 1, 1] if model_cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    
    if(split == 'train'):
        for d_idx, data in enumerate(tqdm.tqdm(dataloader)):

            dir_path = data.pop('dir')
            frame_idxs = data.pop('frame_idxs')
            data = {k: v.to(device) for k, v in data.items()}
            new_folder_path = os.path.join(dir_path[0], "depth_images")

            if os.path.exists(new_folder_path):
                # Delete everything inside the folder
                for filename in os.listdir(new_folder_path):
                    file_path = os.path.join(new_folder_path, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory and its contents
            os.makedirs(new_folder_path, exist_ok=True)



            rot_transform_quats = data["source_cv2wT_quat"][:, :model_cfg.data.input_images]

            if model_cfg.data.category == "hydrants" or model_cfg.data.category == "teddybears":
                focals_pixels_pred = data["focals_pixels"][:, :model_cfg.data.input_images, ...]
            else:
                focals_pixels_pred = None

            if model_cfg.data.origin_distances:
                input_images = torch.cat([data["gt_images"][:, :model_cfg.data.input_images, ...],
                                        data["origin_distances"][:, :model_cfg.data.input_images, ...]],
                                        dim=2)
            else:
                input_images = data["gt_images"][:, :model_cfg.data.input_images, ...]

            example_id = dataloader.dataset.get_example_id(d_idx)

            # batch has length 1, the first image is conditioning
            reconstruction = model(input_images,
                                data["view_to_world_transforms"][:, :model_cfg.data.input_images, ...],
                                rot_transform_quats,
                                focals_pixels_pred)
            
            depth_img = reconstruction.pop('depth')
            depth_img = depth_img.squeeze(0)  # Remove batch dimension


            depth_image_filename = os.path.join(new_folder_path, f"depth_image_64.png")

            # Save the depth image with the correct frame number
            plt.imsave(depth_image_filename, depth_img, cmap='gray')
    else:
        for d_idx, data in enumerate(tqdm.tqdm(dataloader)):
            dir_path = data.pop('dir')
            frame_idxs = data.pop('frame_idxs')
            data = {k: v.to(device) for k, v in data.items()}
            new_folder_path = os.path.join(dir_path[0], "depth_images")

            os.makedirs(new_folder_path, exist_ok=True)

            frame_idxs = frame_idxs.squeeze(0)
            for r_idx in range(data["gt_images"].shape[1]):
                # batch has length 1, the first image is conditioning
                rot_transform_quats = data["source_cv2wT_quat"][:, r_idx:r_idx + 1]

                if model_cfg.data.category == "hydrants" or model_cfg.data.category == "teddybears":
                    focals_pixels_pred = data["focals_pixels"][:, r_idx:r_idx + 1, ...]
                else:
                    focals_pixels_pred = None

                if model_cfg.data.origin_distances:
                    input_images = torch.cat([data["gt_images"][:, r_idx:r_idx + 1, ...],
                                            data["origin_distances"][:, r_idx:r_idx + 1, ...]],
                                            dim=2)
                else:
                    input_images = data["gt_images"][:, r_idx:r_idx + 1, ...]

                example_id = dataloader.dataset.get_example_id(d_idx)



                reconstruction = model(input_images,
                                    data["view_to_world_transforms"][:, r_idx:r_idx + 1, ...],
                                    rot_transform_quats,
                                    focals_pixels_pred)
                
                
                # Create the directory and save the depth image there

                # Iterate over all the depth maps in the reconstruction
                depth_img = reconstruction.pop('depth')
                depth_img = depth_img.squeeze(0)  # Remove batch dimension

                frame_idx_tensor = frame_idxs[r_idx]
                frame_idx = frame_idx_tensor.item()  # Convert the first value to an integer
                depth_image_filename = os.path.join(new_folder_path, f"depth_image_{frame_idx}.png")

                # Save the depth image with the correct frame number
                plt.imsave(depth_image_filename, depth_img, cmap='gray')
        
    return None

@torch.no_grad()
def main(dataset_name, experiment_path, device_idx, split='test', save_vis=0, out_folder=None):
    
    # set device and random seed
    device = torch.device("cuda:{}".format(device_idx))
    torch.cuda.set_device(device)

    if args.experiment_path is None:
        cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                 filename="config_{}.yaml".format(dataset_name))
        if dataset_name in ["gso", "objaverse"]:
            model_name = "latest"
        else:
            model_name = dataset_name
        model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                            filename="model_{}.pth".format(model_name))
        
    else:
        cfg_path = os.path.join(experiment_path, ".hydra", "config.yaml")
        model_path = os.path.join(experiment_path, "model_latest.pth")
    
    # load cfg
    training_cfg = OmegaConf.load(cfg_path)

    # check that training and testing datasets match if not using official models 
    if args.experiment_path is not None:
        if dataset_name == "gso":
            # GSO model must have been trained on objaverse
            assert training_cfg.data.category == "objaverse", "Model-dataset mismatch"
        else:
            assert training_cfg.data.category == dataset_name, "Model-dataset mismatch"

    # load model
    model = GaussianSplatPredictor(training_cfg)
    ckpt_loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model = model.to(device)
    model.eval()

    # override dataset in cfg if testing objaverse model
    if training_cfg.data.category == "objaverse" and split in ["test", "vis"]:
        training_cfg.data.category = "gso"
    # instantiate dataset loader
    dataset = get_dataset(training_cfg, split)
    # print(f'{dataset}')
    # check_dataset(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            persistent_workers=True, pin_memory=True, num_workers=1)
    # print(f'{dataloader}')
    # check_dataloader(dataloader)
    scores = generate_depth_images(model, dataloader, device, training_cfg, save_vis=save_vis, out_folder=out_folder,split='test')

    return scores


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('dataset_name', type=str, help='Dataset to evaluate on', 
                        choices=['objaverse', 'gso', 'cars', 'chairs', 'hydrants', 'teddybears', 'nmr'])
    parser.add_argument('--experiment_path', type=str, default=None, help='Path to the parent folder of the model. \
                        If set to None, a pretrained model will be downloaded')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'vis', 'train'],
                        help='Split to evaluate on (default: test). \
                        Using vis renders loops and does not return scores - to be used for visualisation. \
                        You can also use this to evaluate on the training or validation splits.')
    parser.add_argument('--out_folder', type=str, default='out', help='Output folder to save renders (default: out)')
    parser.add_argument('--save_vis', type=int, default=0, help='Number of examples for which to save renders (default: 0)')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    dataset_name = args.dataset_name
    print("Extracting depth images on dataset {}".format(dataset_name))
    experiment_path = args.experiment_path
    if args.experiment_path is None:
        print("Will load a model released with the paper.")
    else:
        print("Loading a local model according to the experiment path")
    split = args.split

    out_folder = args.out_folder
    save_vis = args.save_vis


    scores = main(dataset_name, experiment_path, 0, split=split, save_vis=save_vis, out_folder=out_folder)
