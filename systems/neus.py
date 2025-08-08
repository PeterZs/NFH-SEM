import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.utils import parse_optimizer
from systems.criterions import PSNR, binary_cross_entropy

import numpy as np
import cv2
import os

def compute_bsd_image(pred_normal, bsd_d, bsd_c, inv_rot, Q_angles, seg_t = 60):
    if pred_normal.ndim == 2 and inv_rot.ndim == 3:
        pred_normal = pred_normal.unsqueeze(-1)
        pred_normal = torch.bmm(inv_rot, pred_normal).squeeze(-1)
    else:
        pred_normal = torch.matmul(pred_normal, inv_rot.T)  # [N, 3] x [3, 3]^T -> [N, 3]
    
    pred_normal[..., 1] *= -1.0
    pred_normal[..., 2] *= -1.0

    pred_normal = F.normalize(pred_normal, p=2, dim=-1)

    theta = torch.atan2(pred_normal[..., 1], (pred_normal[..., 0]+1e-6))
    tan_phi = torch.sqrt(pred_normal[..., 0]**2 + pred_normal[..., 1]**2) / (pred_normal[..., 2]+1e-6)
    sin_phi = torch.sqrt(pred_normal[..., 0]**2 + pred_normal[..., 1]**2) / torch.sqrt(pred_normal[..., 0]**2 + pred_normal[..., 1]**2 + pred_normal[..., 2]**2+1e-6)
    cos_phi = pred_normal[..., 2] / torch.sqrt(pred_normal[..., 0]**2 + pred_normal[..., 1]**2 + pred_normal[..., 2]**2+1e-6)

    phi = torch.atan(tan_phi)
    seg_t_arc = np.pi * seg_t / 180.0
    mask = torch.abs(phi) < seg_t_arc
    tan_phi = torch.clip(tan_phi, np.tan(-1*seg_t_arc), np.tan(seg_t_arc))
    f_poly = 1 + bsd_d[4]*phi + bsd_d[5]*phi**2 + bsd_d[6]*phi**3 + bsd_d[7]*phi**4 

    pred_bsd_images = []
    for i in range(4):
        theta_i = Q_angles[i] * torch.pi / 180
        pred_bsd_images.append(
            bsd_d[i] * (f_poly * sin_phi * torch.cos(theta - theta_i))
            + bsd_c[4 + i] * cos_phi * f_poly
            + bsd_c[i]
        )

    pred_bsd_image = torch.stack(pred_bsd_images, dim=-1)

    mask = mask.unsqueeze(-1).repeat(1, 4)

    return pred_bsd_image, mask

def organize_bsd_param(bsd_c, bsd_d):
    param_type = 'Unknown'
    if bsd_d.ndim == 0: # tan1
        bsd_d = bsd_d.repeat(4)
        bsd_c = bsd_c.repeat(4)
        param_type = 'Tan'
    elif bsd_d.ndim == 1 and bsd_d.shape[0] == 4:
        param_type = 'Tan'
    elif bsd_d.ndim == 1 and bsd_d.shape[0] == 5: # tan2
        bsd_c = torch.stack([bsd_c[0], bsd_c[0], bsd_c[0], bsd_c[0], bsd_c[1], bsd_c[1], bsd_c[1], bsd_c[1]])
        bsd_d = torch.stack([bsd_d[0], bsd_d[0], bsd_d[0], bsd_d[0], bsd_d[1], bsd_d[2], bsd_d[3], bsd_d[4]])
        param_type = 'Poly'
    elif bsd_d.ndim == 1 and bsd_d.shape[0] == 8:
        param_type = 'Poly'
    return bsd_c, bsd_d, param_type

def bsd_loss(pred_normal, bsd_d, bsd_c, bsd_image, bsd_mask, inv_rot, consider_gray_scale, bsd_mask_type, adaptive_mask_threshold, fg_mask, Q_angles):

    bsd_c, bsd_d, param_type = organize_bsd_param(bsd_c, bsd_d)
    pred_bsd_image, seg_mask = compute_bsd_image(pred_normal, bsd_d, bsd_c, inv_rot, Q_angles)

    fg_mask = fg_mask.unsqueeze(1).repeat(1, 4)
    final_mask = fg_mask

    if bsd_mask_type == 'Empty':
        final_mask = final_mask & seg_mask & bsd_mask
        loss = torch.abs(pred_bsd_image[final_mask].squeeze() - bsd_image[final_mask].squeeze())
    elif bsd_mask_type == 'Predefined':
        final_mask = final_mask & bsd_mask
        loss = torch.abs(pred_bsd_image[final_mask].squeeze() - bsd_image[final_mask].squeeze())
    elif bsd_mask_type == 'Adaptive':
        error_map = torch.abs(pred_bsd_image - bsd_image)
        adapt_mask = error_map < (bsd_d[0:4] * adaptive_mask_threshold)
        final_mask = final_mask & adapt_mask & seg_mask & bsd_mask
        loss = torch.abs(pred_bsd_image[final_mask].squeeze() - bsd_image[final_mask].squeeze())
    if consider_gray_scale:
        loss = torch.where(loss < 0.5/255.0, torch.zeros_like(loss), loss)
        loss = torch.mean(loss)
    else:
        loss = torch.mean(loss)

    return loss


def depth_loss(pred_depth, target_depth, depth_confidence, use_depth_confidence=False):
    valid_mask = ~torch.isnan(target_depth)

    if use_depth_confidence:
        loss = depth_confidence[valid_mask].squeeze() * torch.abs(pred_depth[valid_mask].squeeze() - target_depth[valid_mask].squeeze())
    else:
        loss = torch.abs(pred_depth[valid_mask].squeeze() - target_depth[valid_mask].squeeze())
    return torch.mean(loss) 

@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.dataset.enable_virtual_frame:
                end_idx = len(self.dataset.all_images)
            else:
                end_idx = self.dataset.real_frame_num
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, end_idx, size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, end_idx, size=(1,), device=self.dataset.all_images.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            inv_rot = self.dataset.all_inv_rot[index] 
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
            normal = self.dataset.all_normals[index, y, x].view(-1, self.dataset.all_normals.shape[-1]).to(self.rank)
            normal_mask = self.dataset.all_normals_masks[index, y, x].view(-1).to(self.rank)
            depth = self.dataset.all_depth_imgs[index, y, x].view(-1).to(self.rank)
            depth_confidence = self.dataset.all_depth_confidence[index, y, x].view(-1).to(self.rank)
            bsd_image = self.dataset.all_bsd_images[index, y, x].view(-1, self.dataset.all_bsd_images.shape[-1]).to(self.rank)
            bsd_mask = self.dataset.all_bsd_masks[index, y, x].view(-1, self.dataset.all_bsd_masks.shape[-1]).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            inv_rot = self.dataset.all_inv_rot[index][0] 
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
            normal = self.dataset.all_normals[index].view(-1, self.dataset.all_normals.shape[-1]).to(self.rank)
            normal_mask = self.dataset.all_normals_masks[index].view(-1).to(self.rank)
            depth = self.dataset.all_depth_imgs[index].view(-1).to(self.rank)
            depth_confidence = self.dataset.all_depth_confidence[index].view(-1).to(self.rank)
            bsd_image = self.dataset.all_bsd_images[index].view(-1, self.dataset.all_bsd_images.shape[-1]).to(self.rank)
            bsd_mask = self.dataset.all_bsd_masks[index].view(-1, self.dataset.all_bsd_masks.shape[-1]).to(self.rank)


        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'normal': normal,
            'fg_mask': fg_mask,
            'inv_rot': inv_rot,
            'normal_mask': normal_mask,
            'depth': depth,
            'depth_confidence': depth_confidence,
            'bsd_image': bsd_image,
            'bsd_mask': bsd_mask
        })      
    
    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.

        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)    

        loss_depth = depth_loss(out['depth_full'][out['rays_valid_full'][...,0]], batch['depth'][out['rays_valid_full'][...,0]], batch['depth_confidence'][out['rays_valid_full'][...,0]], self.config.system.loss.use_depth_confidence)
        self.log('train/loss_depth', loss_depth, prog_bar=True)
        loss += loss_depth * self.C(self.config.system.loss.lambda_depth_mse)

        if self.config.system.loss.bsd_mask_type == 'Adaptive':
            if self.global_step >= self.config.system.loss.adaptive_start_step:
                curr_mask_type = 'Adaptive'
            else:
                curr_mask_type = 'Empty'
        else:
            curr_mask_type = self.config.system.loss.bsd_mask_type
        bsd_c, bsd_d = self.model.bsd_network()
        loss_bsd_l1 = bsd_loss(out['comp_normal'][out['rays_valid_full'][...,0]], bsd_d, bsd_c, batch['bsd_image'][out['rays_valid_full'][...,0]], batch['bsd_mask'][out['rays_valid_full'][...,0]], batch['inv_rot'][out['rays_valid_full'][...,0]].squeeze(), self.config.system.loss.consider_gray_scale, curr_mask_type, self.config.system.loss.adaptive_mask_threshold, batch['fg_mask'][out['rays_valid_full'][...,0]].squeeze(), self.dataset.Q_angles)
        bsd_c, bsd_d, param_type = organize_bsd_param(bsd_c, bsd_d)

        for name, value in zip(
            ['c1', 'c2', 'c3', 'c4', 'e1', 'e2', 'e3', 'e4'],
            bsd_c
        ):
            self.log(name, value, prog_bar=True)

        for name, value in zip(
            ['d1', 'd2', 'd3', 'd4', 'p1', 'p2', 'p3', 'p4'],
            bsd_d
        ):
            self.log(name, value, prog_bar=False)

        if self.config.system.loss.lambda_use_bsd_reg: 
            loss_reg_bsd_param = (torch.var(bsd_d[0:4]) + torch.var(bsd_c[0:4]) + torch.var(bsd_c[4:8])) * self.config.system.loss.lambda_reg 
            self.log('train/loss_reg_bsd_param', loss_reg_bsd_param, prog_bar=True)
            if self.global_step >= self.config.model.bsd.warmup_steps:
                loss += loss_reg_bsd_param

        self.log('train/loss_bsd', loss_bsd_l1, prog_bar=True)
        if self.global_step == self.config.model.bsd.warmup_steps:
            self.trainer.optimizers[0].param_groups[2]['lr'] = self.config.system.loss.normal_lr
        if self.global_step >= self.config.model.bsd.warmup_steps:
            loss += loss_bsd_l1 * self.C(self.config.system.loss.lambda_normal) 
        else:
            loss += loss_bsd_l1 * 0.0

        if self.global_step == self.config.system.loss.virtual_close_step:
            print('Close virtual frame, Real Num:', self.dataset.real_frame_num)
            self.dataset.enable_virtual_frame = False


        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=False)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=False)

        return {
            'loss': loss
        }
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        valid_mask = batch['fg_mask'].view(H, W, 3).to(self.rank)
        all_normal = out['comp_normal'].view(H, W, 3).to(self.rank)
        valid_normal = torch.zeros_like(all_normal).to(self.rank)
        valid_normal[valid_mask] = all_normal[valid_mask] 
        bsd_save_name = f"it{self.global_step}-{batch['index'][0].item()}bsd.png"
        self.save_bsd_images(out, batch, bsd_save_name)
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    def save_bsd_images(self, out, batch, bsd_save_name):
        W, H = self.dataset.img_wh
        bsd_c, bsd_d = self.model.bsd_network()
        bsd_c, bsd_d, param_type = organize_bsd_param(bsd_c, bsd_d)
        valid_mask = batch['fg_mask'].to(self.rank)
        all_normal = out['comp_normal'].to(self.rank)
        valid_normal = torch.zeros_like(all_normal).to(self.rank)
        valid_normal[valid_mask] = all_normal[valid_mask] 

        pred_bsd_images, seg_mask = compute_bsd_image(valid_normal.to(bsd_d.device), bsd_d, bsd_c, batch['inv_rot'], self.dataset.Q_angles)
        pred_bsd_images = pred_bsd_images.view(H, W, 4).cpu().numpy()
        ori_bsd_images = batch['bsd_image'].view(H, W, 4).cpu().numpy()
        bsd_d_np = bsd_d.cpu().numpy()
        adaptive_mask_threshold = float(self.config.system.loss.adaptive_mask_threshold)

        t_mask_images = batch['bsd_mask'].view(H, W, 4).cpu().numpy()

        rows = []
        for i in range(4):  
            ori_img = (ori_bsd_images[:, :, i] * 255).astype(np.uint8)
            pred_img = (np.clip(pred_bsd_images[:, :, i], 0.0, 1.0) * 255).astype(np.uint8)

            error_map = np.abs(ori_bsd_images[:, :, i] - pred_bsd_images[:, :, i])
            error_img = (np.clip(error_map, 0.0, 1.0) * 255).astype(np.uint8)
            error_img = cv2.cvtColor(error_img, cv2.COLOR_GRAY2BGR)

            ori_img_rgb = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
            pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)

            mask_img = np.where(error_map > bsd_d_np[i]*adaptive_mask_threshold, 255, 0).astype(np.uint8)
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

            t_mask_image_rgb = cv2.cvtColor((t_mask_images[:, :, i] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            row = np.hstack((ori_img_rgb, pred_img_rgb, error_img, mask_img))
            rows.append(row)

        final_image = np.vstack(rows)

        output_path = self.get_save_path(bsd_save_name)
        cv2.imwrite(output_path, final_image)
    
    def save_normal_depth_npz(self, out, batch):
        W, H = self.dataset.img_wh
        pred_normal = out['comp_normal'].view(H, W, 3).cpu().numpy()
        pred_depth = out['depth'].view(H, W).cpu().numpy()
        normal_depth_save_folder = self.get_save_path(f"Predict_Normal_Depth/")
        np.savez(os.path.join(normal_depth_save_folder, "{:04d}.npz".format(batch['index'][0].item())), normal_map=pred_normal, depth_map=pred_depth)

    
    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        valid_mask = batch['fg_mask'].view(H, W).to(self.rank)
        all_normal = out['comp_normal'].view(H, W, 3).to(self.rank)
        valid_normal = torch.zeros_like(all_normal).to(self.rank)
        valid_normal[valid_mask] = all_normal[valid_mask] 
        #bsd_save_name = "4Q-BSE.png"
        #self.save_bsd_images(out, batch, bsd_save_name)
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            
            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"{self.config.tag}.obj",
            **mesh
        )        
