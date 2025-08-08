import os
import json
import math
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn as nn
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank

import imageio 
import cv2

def rotate_gradients(gradient_x, gradient_y, angle):
    angle_rad = np.deg2rad(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    rotated_gradient_x = cos_theta * gradient_x - sin_theta * gradient_y
    rotated_gradient_y = sin_theta * gradient_x + cos_theta * gradient_y

    return rotated_gradient_x, rotated_gradient_y


def read_bse_images(root_dir, idx, is_3Q_BSE, t=0.85):
    images = {}
    for detector in ['Q1', 'Q2', 'Q3', 'Q4']:
        if is_3Q_BSE and detector == 'Q4':
            images[detector] = np.zeros_like(np.array(Image.open(os.path.join(root_dir, 'Q1', '{:03d}.png'.format(idx+1))).convert("L")), dtype=np.float32)
            continue
        img_path = os.path.join(root_dir, detector, '{:03d}.png'.format(idx+1))
        images[detector] = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        images[detector] = cv2.GaussianBlur(np.array(images[detector]), (3, 3), 1.0)
    gradient_y = (images["Q1"] - images["Q3"]) / (images["Q1"] + images["Q3"] + 1e-6)
    gradient_x = (images["Q2"] - images["Q4"]) / (images["Q2"] + images["Q4"] + 1e-6)
    gradient_x, gradient_y = rotate_gradients(gradient_x, gradient_y, 25.0)

    # merge four detectors as bsd 4 channel image
    bsd_image = np.stack([images["Q1"], images["Q2"], images["Q3"], images["Q4"]], axis=-1)
    bsd_image = bsd_image / 255.0

    normal_mask = np.ones_like(images["Q1"], dtype=bool)

    bsd_mask = np.ones_like(bsd_image, dtype=bool)

    h, w = gradient_x.shape
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    scale = 1.0 
    normal_map[..., 0] = -gradient_x*scale  
    normal_map[..., 1] = -gradient_y*scale 
    normal_map[..., 2] = 1.0 

    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map /= (norm + 1e-6)

    return normal_map, normal_mask, bsd_image, bsd_mask

class SEMDatasetBase():

    def correct_depth(self, depth_img, intrinsic_mat):
        fx = intrinsic_mat[0, 0]
        fy = intrinsic_mat[1, 1]
        cx = intrinsic_mat[0, 2]
        cy = intrinsic_mat[1, 2]
        W = depth_img.shape[1]
        H = depth_img.shape[0]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.flatten()
        v = v.flatten()
        depth = depth_img.flatten()

        # depth is nan 
        invalid_mask = np.isnan(depth)
        depth[invalid_mask] = 0

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        depth_corrected = np.sqrt(x**2 + y**2 + z**2)
        depth_corrected[invalid_mask] = np.nan

        depth_corrected = depth_corrected.reshape(H, W)
        return depth_corrected


    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = False
        self.apply_mask = False

        self.w, self.h = self.config.real_img_wh
        self.img_wh = (self.w, self.h)

        camera_folder_name = 'camera' 
        self.data_num = len([name for name in os.listdir(os.path.join(self.config.root_dir, camera_folder_name)) if name.endswith('.npz')])
        meta_data = np.load(os.path.join(self.config.root_dir, camera_folder_name, '001.npz'))
        intrinsic_mat = meta_data['intrinsic_mat']
        self.focal = intrinsic_mat[0, 0]

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.w, self.h, intrinsic_mat[0, 0], intrinsic_mat[1, 1], intrinsic_mat[0, 2], intrinsic_mat[1, 2]).to(self.rank) # (h, w, 3)
        self.near, self.far = 15.0, 22.0

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_normals, self.all_inv_rot, self.all_normals_masks, self.all_depth_imgs = [], [], [], [], [], [], []
        self.all_depth_confidence = []
        self.all_bsd_images, self.all_bsd_masks = [], []

        self.real_frame_num = 0
        self.enable_virtual_frame = True

        self.Q_angles = [self.config.Q1, self.config.Q2, self.config.Q3, self.config.Q4]

        if not self.config.use_virtual:
            self.enable_virtual_frame = False
            self.data_num = len([name for name in os.listdir(os.path.join(self.config.root_dir, 'Q1')) if name.endswith('.png')])

        if self.split == 'test': 
            self.data_num = 1

        for i in range(self.data_num):
            meta_data = np.load(os.path.join(self.config.root_dir, camera_folder_name, '{:03d}.npz'.format(i+1)))
            pose = meta_data['extrinsic_mat']
            pose = np.concatenate([pose, np.array([[0,0,0,1]])], 0)
            inv_rot = pose[:3, :3]
            inv_rot = torch.from_numpy(inv_rot)
            pose = np.linalg.inv(pose) 
            pose = torch.from_numpy(pose[:3, :4])
            self.all_c2w.append(pose)

            img_path = os.path.join(self.config.root_dir, 'Q1', '{:03d}.png'.format(i+1))
            is_virtual_frame = False
            if not os.path.exists(img_path):
                is_virtual_frame = True
                img = torch.zeros((self.h, self.w, 3))
                fg_mask = torch.zeros_like(img[..., -1], dtype=bool)
            else:
                is_virtual_frame = False
                self.real_frame_num += 1
                img = torch.zeros((self.h, self.w, 3))

                fg_mask = torch.ones_like(img[..., -1], dtype=bool)
            self.all_images.append(img[...,:3])

            depth_img = meta_data['depth_map']
            depth_img = self.correct_depth(depth_img, intrinsic_mat)
            depth_img = torch.from_numpy(depth_img)
            self.all_depth_imgs.append(depth_img)

            if self.config.mask_invalid_depth:
                depth_valid_mask = ~torch.isnan(depth_img)
                fg_mask = fg_mask & depth_valid_mask
            self.all_fg_masks.append(fg_mask)

            depth_confidence = meta_data['conf_map']
            depth_confidence = (depth_confidence*0.99)+0.01
            if is_virtual_frame:
                depth_confidence = torch.from_numpy(depth_confidence*self.config.virtual_depth_confidence_scale)
            else:
                depth_confidence = torch.from_numpy(depth_confidence)
            self.all_depth_confidence.append(depth_confidence)
            
            if is_virtual_frame:
                normal_map = np.zeros((self.h, self.w, 3), dtype=np.float32)
                normal_mask = np.zeros((self.h, self.w), dtype=bool)
                bsd_image = np.zeros((self.h, self.w, 4), dtype=np.float32)
                bsd_mask = np.zeros((self.h, self.w, 4), dtype=bool)
            else:
                normal_map, normal_mask, bsd_image, bsd_mask = read_bse_images(self.config.root_dir, i, self.config.is_3Q_BSE)
                normal_mask_im = normal_mask.astype(np.uint8)
                normal_mask_im = Image.fromarray(normal_mask_im*255)

            normal_map = torch.from_numpy(normal_map)
            self.all_normals.append(normal_map)

            bsd_mask = np.ones_like(bsd_mask, dtype=bool)

            if self.config.is_3Q_BSE:
                bsd_mask[..., 3] = 0

            normal_mask = torch.from_numpy(normal_mask)
            self.all_normals_masks.append(normal_mask)

            self.all_inv_rot.append(inv_rot)

            self.all_bsd_images.append(torch.from_numpy(bsd_image))
            self.all_bsd_masks.append(torch.from_numpy(bsd_mask))
        
        self.all_c2w, self.all_images, self.all_fg_masks, self.all_normals, self.all_inv_rot, self.all_normals_masks, self.all_depth_imgs, self.all_depth_confidence, self.all_bsd_images, self.all_bsd_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).to(self.rank), \
            torch.stack(self.all_normals, dim=0).float().to(self.rank), \
            torch.stack(self.all_inv_rot, dim=0).float().to(self.rank), \
            torch.stack(self.all_normals_masks, dim=0).to(self.rank), \
            torch.stack(self.all_depth_imgs, dim=0).to(self.rank), \
            torch.stack(self.all_depth_confidence, dim=0).to(self.rank), \
            torch.stack(self.all_bsd_images, dim=0).to(self.rank), \
            torch.stack(self.all_bsd_masks, dim=0).to(self.rank)
        

class SEMDataset(Dataset, SEMDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class SEMIterableDataset(IterableDataset, SEMDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('sem')
class SEMDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = SEMIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = SEMDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = SEMDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = SEMDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       