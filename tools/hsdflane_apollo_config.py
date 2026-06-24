"""
hsdflane_apollo_config.py 
=========================================================
Split = ["standard", "rare_subset", "illus_chg"]   
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from loader.bev_road.apollo_data import (
    Apollo_dataset_with_offset,
    Apollo_dataset_with_offset_val,
)
from models.model.hsdflane import HSDFLane
from models.loss import IoULoss, NDPushPullLoss, GaussianFocalLoss, SDFFieldLoss, CustomSmoothL1Loss
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APOLLO_ROOT = os.path.join(os.path.dirname(_REPO_ROOT), 'dataset', 'Apollo_Sim_3D_Lane_Release')

# Choose Split: "standard" | "rare_subset" | "illus_chg"
SPLIT_TYPE = "standard"

_SPLIT_ROOT = os.path.join(APOLLO_ROOT, 'splits', SPLIT_TYPE)

train = {
    "json": os.path.join(_SPLIT_ROOT, 'train.json'),
    "base": APOLLO_ROOT,
}
val = {
    "json": os.path.join(_SPLIT_ROOT, 'val.json'),
    "base": APOLLO_ROOT,
}

print(f"[hsdflane_apollo_config] SPLIT_TYPE={SPLIT_TYPE}")
print(f"  train → {train['json']}")
print(f"  val   → {val['json']}")

model_save_path = f"./hsdflane_apollo_{SPLIT_TYPE}"

input_shape     = (600, 800)
output_2d_shape = (144, 256)

x_range         = (3, 103)
y_range         = (-12, 12)
meter_per_pixel = 0.5
bev_shape = (
    int((x_range[1] - x_range[0]) / meter_per_pixel),   # 200
    int((y_range[1] - y_range[0]) / meter_per_pixel),   #  48
)


loader_args     = dict(batch_size=4, num_workers=12)
val_loader_args = dict(batch_size=4, num_workers=4)

# Validation post-processing parameters
post_conf = -1.7
post_emb_margin = 6.0
post_min_cluster_size = 15


def model():
    return HSDFLane(
        bev_shape=bev_shape,
        image_shape=input_shape,
        output_2d_shape=output_2d_shape,
        train=True,
        min_spacing=1.5,
        base_tau=0.3,
        min_angle=-5.0,
        max_angle=5.0,
    )


epochs = 100
optimizer = AdamW
optimizer_params = dict(
    lr=5e-4, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False,
)
scheduler = CosineAnnealingLR


def train_dataset():
    train_trans = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.MotionBlur(p=0.2),
        A.RandomBrightnessContrast(),
        A.ColorJitter(p=0.1),
        A.Normalize(),
        ToTensorV2(),
    ])
    return Apollo_dataset_with_offset(
        data_json_path=train["json"],
        dataset_base_dir=train["base"],
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        data_trans=train_trans,
        output_2d_shape=output_2d_shape,
        input_shape=input_shape,
        heatmap_sigma=2.0,
        heatmap_radius=6,
    )


def val_dataset():
    val_trans = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(),
        ToTensorV2(),
    ])
    return Apollo_dataset_with_offset_val(
        data_json_path=val["json"],
        dataset_base_dir=val["base"],
        data_trans=val_trans,
        x_range=x_range,
        y_range=y_range,
        meter_per_pixel=meter_per_pixel,
        input_shape=input_shape,
    )

class Combine_Model_and_Loss(nn.Module):
    def __init__(self, model, heatmap_loss_weight=5.0, eikonal_weight=0.1):
        super().__init__()
        self.model = model
        self.heatmap_loss_weight = heatmap_loss_weight
        self.return_loss_details = False

        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.emb_loss = NDPushPullLoss(1.0, 1.0, 1.0, 5.0, 200)
        self.l1_loss = CustomSmoothL1Loss()
        self.sdf_loss = SDFFieldLoss(eikonal_weight=eikonal_weight)
        self.heatmap_focal = GaussianFocalLoss(alpha=2.0, beta=4.0)

    def forward(self, inputs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None,
                image_gt_segment=None, image_gt_instance=None, gt_height=None,
                intrinsic=None, road2cam=None, train=True,
                lane_heatmap_gt=None):

        if train:
            (pred, emb, offset_y), heightmap, (pred_2d, emb_2d), \
                lane_heatmap, sdf_pred, z_sampled = self.model(inputs, intrinsic, road2cam)
        else:
            result = self.model(inputs, intrinsic, road2cam)
            pred, emb, offset_y = result[0]
            heightmap = result[1]
            if heightmap.dim() == 3:
                heightmap = heightmap.unsqueeze(1)
            return pred, emb, offset_y, heightmap, None

        gt_heightmap = gt_height[:, 0:1, :, :]
        gt_heightmask = gt_height[:, 1:2, :, :]

        if heightmap.dim() == 3:
            heightmap = heightmap.unsqueeze(1)

        loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)
        loss_emb = self.emb_loss(emb, gt_instance)
        loss_offset_raw = F.binary_cross_entropy_with_logits(offset_y, gt_offset_y, weight=gt_seg)

        loss_total = 5 * loss_seg + loss_emb
        loss_offset = 60 * loss_offset_raw

        loss_rendering = self.l1_loss(heightmap, gt_heightmap, gt_heightmask)
        sdf_valid = self.model._sdf_valid_mask
        loss_sdf_field = self.sdf_loss(sdf_pred, z_sampled, gt_heightmap, gt_heightmask, sdf_valid)
        loss_height = 10 * (1.0 * loss_rendering + 1.0 * loss_sdf_field)

        loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(
            torch.sigmoid(pred_2d), image_gt_segment)
        loss_emb_2d = self.emb_loss(emb_2d, image_gt_instance)
        loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d

        loss_heatmap = self.heatmap_focal(lane_heatmap, lane_heatmap_gt)
        loss_height = loss_height + self.heatmap_loss_weight * loss_heatmap

        if self.return_loss_details:
            loss_details = {
                "seg_raw": loss_seg,
                "emb_raw": loss_emb,
                "offset_raw": loss_offset_raw,
                "seg2d_raw": loss_seg_2d,
                "emb2d_raw": loss_emb_2d,
                "rendering_raw": loss_rendering,
                "sdf_field_raw": loss_sdf_field,
                "heatmap_raw": loss_heatmap,
                "bev_weighted": loss_total,
                "offset_weighted": loss_offset,
                "2d_weighted": loss_total_2d,
                "height_weighted": loss_height,
            }
            return pred, loss_total, loss_offset, loss_total_2d, loss_height, loss_details

        return pred, loss_total, loss_offset, loss_total_2d, loss_height
