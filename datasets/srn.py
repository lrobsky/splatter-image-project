import glob
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import re
from .dataset_readers import readCamerasFromTxt
from utils.general_utils import PILtoTorch, matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World

from .shared_dataset import SharedDataset

SHAPENET_DATASET_ROOT = r"SHAPENET_DATASET_ROOT"
 # Change this to your data directory
assert SHAPENET_DATASET_ROOT is not None, "Update the location of the SRN Shapenet Dataset"

class SRNDataset(SharedDataset):
    def __init__(self, cfg,
                 dataset_name="train"):
        super().__init__()
        self.cfg = cfg

        self.dataset_name = dataset_name
        if dataset_name == "vis":
            self.dataset_name = "test"

        self.base_path = os.path.join(SHAPENET_DATASET_ROOT, "srn_{}/{}_{}".format(cfg.data.category,
                                                                                   cfg.data.category,
                                                                                   self.dataset_name))

        is_chair = "chair" in cfg.data.category
        if is_chair and dataset_name == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )


        if cfg.data.subset != -1:
            self.intrins = self.intrins[:cfg.data.subset]

        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
        
        self.imgs_per_obj = self.cfg.opt.imgs_per_obj


        # in deterministic version the number of testing images
        # and number of training images are the same
        if self.cfg.data.input_images == 1:
            self.test_input_idxs = [64]
        elif self.cfg.data.input_images == 2:
            self.test_input_idxs = [64, 128]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.intrins)
    
    def load_example_id(self, example_id, intrin_path,
                        trans = np.array([0.0, 0.0, 0.0]), scale=1.0):
        def extract_number(file_path):
            return int((file_path.split('_'))[-1].split('.')[0])  # Extract the last number in the path
        
        self.dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(self.dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(self.dir_path, "pose", "*")))
        #
        depth_paths = sorted(glob.glob(os.path.join(self.dir_path, "depth_images", "*")), key= extract_number)

        assert len(rgb_paths) == len(pose_paths)


        

        if not hasattr(self, "all_rgbs"):
            self.all_rgbs = {}
            self.all_world_view_transforms = {}
            self.all_view_to_world_transforms = {}
            self.all_full_proj_transforms = {}
            self.all_camera_centers = {}
            self.all_depths = {}

        if example_id not in self.all_rgbs.keys():
            self.all_rgbs[example_id] = []
            self.all_world_view_transforms[example_id] = []
            self.all_full_proj_transforms[example_id] = []
            self.all_camera_centers[example_id] = []
            self.all_view_to_world_transforms[example_id] = []
            self.all_depths[example_id] = []

            cam_infos = readCamerasFromTxt(rgb_paths, pose_paths, [i for i in range(len(rgb_paths))])

            for cam_info in cam_infos:
                R = cam_info.R
                T = cam_info.T

                self.all_rgbs[example_id].append(PILtoTorch(cam_info.image, 
                                                            (self.cfg.data.training_resolution, self.cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :])
                

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
                view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                self.all_world_view_transforms[example_id].append(world_view_transform)
                self.all_view_to_world_transforms[example_id].append(view_world_transform)
                self.all_full_proj_transforms[example_id].append(full_proj_transform)
                self.all_camera_centers[example_id].append(camera_center)


            for i in range(len(depth_paths)):
                depth_image = Image.open(depth_paths[i]).convert('L')  # Open the depth image
                depth_tensor = torch.from_numpy(np.array(depth_image)).unsqueeze(0)  # Convert to tensor and add a channel dimension
                self.all_depths[example_id].append(depth_tensor)  # Append depth image
            

            


            self.all_world_view_transforms[example_id] = torch.stack(self.all_world_view_transforms[example_id])
            self.all_view_to_world_transforms[example_id] = torch.stack(self.all_view_to_world_transforms[example_id])
            self.all_full_proj_transforms[example_id] = torch.stack(self.all_full_proj_transforms[example_id])
            self.all_camera_centers[example_id] = torch.stack(self.all_camera_centers[example_id])
            self.all_rgbs[example_id] = torch.stack(self.all_rgbs[example_id])
            self.all_depths[example_id] = torch.stack(self.all_depths[example_id])

    def get_example_id(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))
        return example_id

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        example_id = os.path.basename(os.path.dirname(intrin_path))

        self.load_example_id(example_id, intrin_path)
        if self.dataset_name == "train":
            frame_idxs = torch.randperm(
                    len(self.all_rgbs[example_id])
                    )[:self.imgs_per_obj]


        else:
            input_idxs = self.test_input_idxs

            frame_idxs = torch.cat([torch.tensor(input_idxs), 
                                    torch.tensor([i for i in range(251) if i not in input_idxs])], dim=0) 


        if self.dataset_name == 'train':
            images_and_camera_poses = {
                "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
                "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
                "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
                "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
                "camera_centers": self.all_camera_centers[example_id][frame_idxs],
                "depths": self.all_depths[example_id][frame_idxs]

            }
        else:
            images_and_camera_poses = {
                "gt_images": self.all_rgbs[example_id][frame_idxs].clone(),
                "world_view_transforms": self.all_world_view_transforms[example_id][frame_idxs],
                "view_to_world_transforms": self.all_view_to_world_transforms[example_id][frame_idxs],
                "full_proj_transforms": self.all_full_proj_transforms[example_id][frame_idxs],
                "camera_centers": self.all_camera_centers[example_id][frame_idxs],
                "depths": self.all_depths[example_id]
            }

        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(images_and_camera_poses["view_to_world_transforms"])

        return images_and_camera_poses