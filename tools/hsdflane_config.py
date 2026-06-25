"""
hsdflane_config.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from loader.bev_road.openlane_data import (
    OpenLane_dataset_with_offset,
    OpenLane_dataset_with_offset_val,
)
from models.model.hsdflane import HSDFLane
from models.loss import IoULoss, NDPushPullLoss, GaussianFocalLoss, SDFFieldLoss, CustomSmoothL1Loss
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.join(os.path.dirname(_REPO_ROOT), 'dataset', 'openlane')

train = {
    "gt": os.path.join(ROOT_DIR, 'training'),
    "images": os.path.join(ROOT_DIR, 'images', 'training'),
    "maps": os.path.join(ROOT_DIR, 'heightmap_training'),
}

val = {
    "gt": os.path.join(ROOT_DIR, 'validation'),
    "images": os.path.join(ROOT_DIR, 'images', 'validation'),
    "maps": os.path.join(ROOT_DIR, 'heightmap_validation'),
}

model_save_path = "./hsdflane"

input_shape = (600, 800)
output_2d_shape = (144, 256)

x_range = (3, 103)
y_range = (-12, 12)
meter_per_pixel = 0.5
bev_shape = (
    int((x_range[1] - x_range[0]) / meter_per_pixel),
    int((y_range[1] - y_range[0]) / meter_per_pixel),
)

loader_args = dict(batch_size=8, num_workers=8)
val_loader_args = dict(batch_size=8, num_workers=8)

# validation post-processing parameters
post_conf = -1.5
post_emb_margin = 6.5
post_min_cluster_size = 10


# ═════════════════════════════════════════════════════════════
#  Model
# ═════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════
#  Optimizer / Scheduler
# ═════════════════════════════════════════════════════════════
epochs = 24
optimizer = AdamW
optimizer_params = dict(
    lr=5e-4, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-2, amsgrad=False,
)
scheduler = CosineAnnealingLR


# ═════════════════════════════════════════════════════════════
#  Datasets
# ═════════════════════════════════════════════════════════════
def train_dataset():
    train_trans = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.MotionBlur(p=0.2),
        A.RandomBrightnessContrast(),
        A.ColorJitter(p=0.1),
        A.Normalize(),
        ToTensorV2(),
    ])
    return OpenLane_dataset_with_offset(
        train["images"], train["gt"], train["maps"],
        x_range, y_range, meter_per_pixel,
        train_trans, input_shape, output_2d_shape,
        heatmap_sigma=2.0, heatmap_radius=6,
    )


def val_dataset():
    trans_image = A.Compose([
        A.Resize(height=input_shape[0], width=input_shape[1]),
        A.Normalize(),
        ToTensorV2(),
    ])
    return OpenLane_dataset_with_offset_val(
        val["images"], val["gt"], val["maps"],
        trans_image,
    )


# ═════════════════════════════════════════════════════════════
#  Combine_Model_and_Loss
# ═════════════════════════════════════════════════════════════
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
