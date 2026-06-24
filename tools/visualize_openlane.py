"""
visualize_openlane.py  -  HSDF_Lane visualization script for OpenLane
======================================================================
Saves three images per sample:
  - heightmap_lane_bev.png     : Heightmap pred / GT in BEV
  - lane_projection_image.png  : Lane pred / GT projected onto the input image
  - lanes_3d_bev.png           : Lane pred / GT as 2x3 panel (3D + BEV)

Usage:
    python tools/visualize_openlane.py \
        --config tools/hsdf_lane_config.py \
        --checkpoint ./checkpoint/latest.pth \
        --save_root ./vis_results \
        --max_samples 10 \
        --sample_interval 5 \
        --gpu 0

    # Visualize a specific segment only:
    python tools/visualize_openlane.py \
        --config tools/hsdf_lane_config.py \
        --checkpoint ./checkpoint/latest.pth \
        --save_root ./vis_results \
        --segment segment-scene-0001 \
        --gpu 0
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import copy
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config_util import load_config_module
from models.util.load_model import load_model


def denormalize_image(tensor_img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor_img.cpu().clone()
    img = img * std + mean
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


class LaneVisualizer:

    def __init__(self, model, save_root='./vis_results',
                 img_h=600, img_w=800,
                 x_range=(3, 103), y_range=(-12, 12),
                 meter_per_pixel=0.5,
                 post_conf=-1.7, post_emb_margin=6.0,
                 post_min_cluster_size=15):
        self.model = model
        self.save_root = save_root
        self.img_h = img_h
        self.img_w = img_w
        self.x_range = x_range
        self.y_range = y_range
        self.meter_per_pixel = meter_per_pixel
        self.post_conf = post_conf
        self.post_emb_margin = post_emb_margin
        self.post_min_cluster_size = post_min_cluster_size

    def visualize(self, original_image_np, sample_name, batch_idx=0,
                  pred_=None, height_pred=None, heightmap_gt=None,
                  intrinsic=None, extrinsic=None, gt_path=None):
        sample_dir = os.path.join(self.save_root, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        lanes_gt_bev, lanes_gt_cam = None, None
        if gt_path is not None and os.path.exists(gt_path):
            lanes_gt_bev, lanes_gt_cam = self._load_gt_lanes(gt_path)

        if pred_ is not None and height_pred is not None:
            lanes_pred = self._postprocess_lanes(pred_, height_pred, batch_idx)

            self._visualize_heightmap_and_lanes(
                height_pred, heightmap_gt, sample_dir, batch_idx)

            if intrinsic is not None and extrinsic is not None:
                self._visualize_lanes_on_image(
                    original_image_np, lanes_pred, lanes_gt_cam,
                    intrinsic, extrinsic, sample_dir, batch_idx)

            self._visualize_lanes_3d_bev(
                lanes_pred, lanes_gt_bev, sample_dir, batch_idx)

    def _postprocess_lanes(self, pred_, height_pred, batch_idx):
        from models.util.cluster import embedding_post
        from models.util.post_process import bev_instance2points_with_offset_z

        seg = pred_[0][batch_idx].unsqueeze(0).detach().cpu()
        emb = pred_[1][batch_idx].unsqueeze(0).detach().cpu()
        offset_y = torch.sigmoid(pred_[2][batch_idx]).unsqueeze(0).detach().cpu()
        height = height_pred[batch_idx, 0].detach().cpu().numpy()

        canvas, ids = embedding_post((seg, emb),
                                     conf=self.post_conf,
                                     emb_margin=self.post_emb_margin,
                                     min_cluster_size=self.post_min_cluster_size,
                                     canvas_color=False)

        lines = bev_instance2points_with_offset_z(
            canvas, max_x=self.x_range[1],
            meter_per_pixal=(self.meter_per_pixel, self.meter_per_pixel),
            offset_y=offset_y[0][0].numpy(),
            Z=height)

        lanes = []
        for lane in lines:
            x_m = np.array(lane[0])
            y_m = np.array(lane[1])
            z_m = np.array(lane[2])
            bev_row = (self.x_range[1] - x_m) / self.meter_per_pixel
            bev_col = (abs(self.y_range[0]) - y_m) / self.meter_per_pixel
            lanes.append({'x': x_m, 'y': y_m, 'z': z_m,
                          'bev_row': bev_row, 'bev_col': bev_col})
        return lanes

    def _load_gt_lanes(self, gt_path):
        cam_representation = np.linalg.inv(
            np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]], dtype=float))

        R_vg = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        R_gc = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

        with open(gt_path, 'r') as f:
            gt = json.load(f)

        cam_w_extrinsics = np.array(gt['extrinsic'])
        cam_extrinsics_pf = copy.deepcopy(cam_w_extrinsics)
        cam_extrinsics_pf[:3, :3] = np.matmul(np.matmul(
            np.matmul(np.linalg.inv(R_vg), cam_extrinsics_pf[:3, :3]),
            R_vg), R_gc)
        cam_extrinsics_pf[0:2, 3] = 0.0
        matrix_lane2pf = cam_extrinsics_pf @ cam_representation

        lanes_bev = []
        lanes_cam = []

        for lane_info in gt['lane_lines']:
            vis = np.array(lane_info['visibility'])
            xyz = np.array(lane_info['xyz']).T
            xyz = xyz[vis == 1.0]
            if xyz.shape[0] < 2:
                continue
            cam_w = np.vstack([xyz.T, np.ones((1, xyz.shape[0]))])
            ego_pf = matrix_lane2pf @ cam_w

            distance = ((ego_pf[1, 0] - ego_pf[1, -1]) ** 2
                        + (ego_pf[0, 0] - ego_pf[0, -1]) ** 2)
            if distance <= 9:
                continue

            x_m = ego_pf[1]
            y_m = -ego_pf[0]
            z_m = ego_pf[2]

            bev_row = (self.x_range[1] - x_m) / self.meter_per_pixel
            bev_col = (abs(self.y_range[0]) - y_m) / self.meter_per_pixel

            lanes_bev.append({'x': x_m, 'y': y_m, 'z': z_m,
                               'bev_row': bev_row, 'bev_col': bev_col})
            lanes_cam.append(cam_w[:3, :])

        return lanes_bev, lanes_cam

    # ------------------------------------------------------------------
    #  heightmap_lane_bev.png
    # ------------------------------------------------------------------
    def _visualize_heightmap_and_lanes(self, height_pred, heightmap_gt,
                                       sample_dir, batch_idx):
        h_pred = height_pred[batch_idx, 0].detach().cpu().numpy()

        if heightmap_gt is not None:
            h_gt = heightmap_gt[batch_idx, 0].cpu().numpy()
            h_mask = heightmap_gt[batch_idx, 1].cpu().numpy()
            h_gt_masked = np.where(h_mask > 0.5, h_gt, np.nan)
            h_pred_masked = np.where(h_mask > 0.5, h_pred, np.nan)
        else:
            h_gt_masked = np.full_like(h_pred, np.nan)
            h_pred_masked = h_pred.copy()

        vmin = np.nanmin([np.nanmin(h_pred), np.nanmin(h_gt_masked)])
        vmax = np.nanmax([np.nanmax(h_pred), np.nanmax(h_gt_masked)])

        fig, axes = plt.subplots(1, 3, figsize=(15, 10))

        im0 = axes[0].imshow(h_pred, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title('Heightmap Pred (full)', fontsize=9)
        axes[0].set_xlabel('Y (lateral)')
        axes[0].set_ylabel('X (longitudinal)')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(h_pred_masked, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title('Heightmap Pred (masked)', fontsize=9)
        axes[1].set_xlabel('Y (lateral)')
        axes[1].set_ylabel('X (longitudinal)')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(h_gt_masked, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        axes[2].set_title('Heightmap GT (masked)', fontsize=9)
        axes[2].set_xlabel('Y (lateral)')
        axes[2].set_ylabel('X (longitudinal)')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'heightmap_lane_bev.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    #  lane_projection_image.png
    # ------------------------------------------------------------------
    def _visualize_lanes_on_image(self, original_image_np, lanes_pred,
                                  lanes_gt_cam, intrinsic, extrinsic,
                                  sample_dir, batch_idx):
        intr = intrinsic[batch_idx].detach().cpu().numpy()
        extr = extrinsic[batch_idx].detach().cpu().numpy()

        # camera-world → standard camera coordinate transform
        matrix_cw2c = np.array([
            [0., -1.,  0., 0.],
            [0.,  0., -1., 0.],
            [1.,  0.,  0., 0.],
            [0.,  0.,  0., 1.]], dtype=float)
        extr_full = matrix_cw2c @ extr

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        axes[0].imshow(original_image_np)
        axes[0].set_title('Lane Pred Projection', fontsize=11)
        axes[0].axis('off')
        colors_pred = plt.cm.cool(np.linspace(0.2, 0.8, max(len(lanes_pred), 1)))
        for li, lane in enumerate(lanes_pred):
            pts_img = self._project_lane_to_image(
                lane['y'], lane['x'], lane['z'],
                extr_full, intr, self.img_h, self.img_w)
            if pts_img is not None and len(pts_img) > 0:
                axes[0].plot(pts_img[:, 0], pts_img[:, 1], '-',
                             color=colors_pred[li], linewidth=2.5, alpha=0.8)

        axes[1].imshow(original_image_np)
        axes[1].set_title('Lane GT Projection', fontsize=11)
        axes[1].axis('off')
        if lanes_gt_cam:
            cw2c_3x3 = matrix_cw2c[:3, :3]
            colors_gt = plt.cm.summer(np.linspace(0.2, 0.8, max(len(lanes_gt_cam), 1)))
            for li, cam_xyz in enumerate(lanes_gt_cam):
                cam_std = cw2c_3x3 @ cam_xyz
                depth = cam_std[2, :]
                valid = depth > 0.1
                pts_2d = intr @ cam_std
                pts_2d = pts_2d[:2, :] / pts_2d[2:, :]
                mask = (valid
                        & (pts_2d[0] >= 0) & (pts_2d[0] < self.img_w)
                        & (pts_2d[1] >= 0) & (pts_2d[1] < self.img_h))
                px, py = pts_2d[0, mask], pts_2d[1, mask]
                if len(px) > 1:
                    axes[1].plot(px, py, '-', color=colors_gt[li],
                                 linewidth=2.5, alpha=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'lane_projection_image.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _project_lane_to_image(y_lateral, x_longitudinal, z_height,
                               extr_full, intr, img_h, img_w):
        N = len(x_longitudinal)
        if N < 2:
            return None
        ones = np.ones(N)
        # model coord: y is left+, so negate to get camera-world lateral
        world = np.stack([-y_lateral, x_longitudinal, z_height, ones], axis=0)
        cam = extr_full @ world
        pts_2d = intr @ cam[:3, :]
        depth = pts_2d[2, :]
        valid = depth > 0.1
        pts_2d = pts_2d[:2, :] / pts_2d[2:, :]
        valid &= ((pts_2d[0] >= 0) & (pts_2d[0] < img_w)
                  & (pts_2d[1] >= 0) & (pts_2d[1] < img_h))
        if valid.sum() < 2:
            return None
        return pts_2d[:, valid].T

    # ------------------------------------------------------------------
    #  lanes_3d_bev.png
    # ------------------------------------------------------------------
    def _visualize_lanes_3d_bev(self, lanes_pred, lanes_gt_bev,
                                sample_dir, batch_idx):
        x_lo, x_hi = self.x_range
        y_lo, y_hi = self.y_range
        z_lo, z_hi = -3.0, 3.0
        x_span = x_hi - x_lo
        y_span = y_hi - y_lo

        PRED_COLOR = '#00BFFF'
        GT_COLOR   = '#FF4500'

        def _filter_lanes(lanes):
            filtered = []
            for lane in lanes:
                x = np.asarray(lane['x'])
                y = np.asarray(lane['y'])
                z = np.asarray(lane['z'])
                mask = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
                if mask.sum() < 2:
                    continue
                fl = dict(lane)
                fl['x'] = x[mask]
                fl['y'] = y[mask]
                fl['z'] = z[mask]
                fl['bev_row'] = np.asarray(lane['bev_row'])[mask]
                fl['bev_col'] = np.asarray(lane['bev_col'])[mask]
                filtered.append(fl)
            return filtered

        def _draw_3d(ax, lane_groups, title):
            for lanes, color, label, ls in lane_groups:
                if not lanes:
                    continue
                label_set = False
                for lane in lanes:
                    lbl = label if not label_set else None
                    ax.plot(np.asarray(lane['x']), np.asarray(lane['y']),
                            np.asarray(lane['z']),
                            linestyle=ls, color=color,
                            markersize=1.5, linewidth=1.8, label=lbl)
                    label_set = True
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('X (fwd, m)', fontsize=8)
            ax.set_ylabel('Y (lat, m)', fontsize=8)
            ax.set_zlabel('Z (m)', fontsize=8)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_zlim(z_lo, z_hi)
            ax.set_box_aspect([x_span * 0.8, y_span, (z_hi - z_lo) * 3])
            ax.view_init(elev=30, azim=200)
            ax.tick_params(labelsize=7)
            if any(len(ls_list) > 0 for ls_list, _, _, _ in lane_groups):
                ax.legend(fontsize=7, loc='upper left')

        def _draw_bev(ax, lane_groups, title):
            ax.set_facecolor('white')
            ax.grid(True, linewidth=0.5, alpha=0.4, color='gray')
            # y is left+, so invert x-axis to match driving perspective
            ax.set_xlim(y_hi, y_lo)
            ax.set_ylim(x_lo, x_hi)
            ax.set_aspect(1.0)
            for lanes, color, label, ls in lane_groups:
                if not lanes:
                    continue
                label_set = False
                for lane in lanes:
                    lbl = label if not label_set else None
                    ax.plot(np.asarray(lane['y']), np.asarray(lane['x']),
                            linestyle=ls, marker='o', color=color,
                            markersize=1.5, linewidth=1.5, label=lbl)
                    label_set = True
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Y  lateral (m)', fontsize=9)
            ax.set_ylabel('X  forward (m)', fontsize=9)
            ax.tick_params(labelsize=7)
            if any(len(ls_list) > 0 for ls_list, _, _, _ in lane_groups):
                ax.legend(fontsize=7, loc='upper left')

        gt_lanes = _filter_lanes(lanes_gt_bev or [])

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(21, 16))

        ax_p3d = fig.add_subplot(2, 3, 1, projection='3d')
        _draw_3d(ax_p3d, [(lanes_pred, PRED_COLOR, 'Pred', '-')], 'Pred  3D')

        ax_g3d = fig.add_subplot(2, 3, 2, projection='3d')
        _draw_3d(ax_g3d, [(gt_lanes, GT_COLOR, 'GT', '--')], 'GT  3D')

        ax_o3d = fig.add_subplot(2, 3, 3, projection='3d')
        _draw_3d(ax_o3d, [(lanes_pred, PRED_COLOR, 'Pred', '-'),
                          (gt_lanes, GT_COLOR, 'GT', '--')], 'Pred + GT  3D')

        ax_pbev = fig.add_subplot(2, 3, 4)
        _draw_bev(ax_pbev, [(lanes_pred, PRED_COLOR, 'Pred', '-')], 'Pred  BEV')

        ax_gbev = fig.add_subplot(2, 3, 5)
        _draw_bev(ax_gbev, [(gt_lanes, GT_COLOR, 'GT', '--')], 'GT  BEV')

        ax_obev = fig.add_subplot(2, 3, 6)
        _draw_bev(ax_obev, [(lanes_pred, PRED_COLOR, 'Pred', '-'),
                            (gt_lanes, GT_COLOR, 'GT', '--')], 'Pred + GT  BEV')

        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, 'lanes_3d_bev.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='HSDF_Lane OpenLane Visualization')
    parser.add_argument('--config', type=str,
                        default='tools/hsdf_lane_config.py',
                        help='Config file path')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint/latest.pth',
                        help='Checkpoint path')
    parser.add_argument('--save_root', type=str,
                        default='./vis_results',
                        help='Visualization output directory')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='Max samples to visualize (0 = all). Ignored if --segment is set.')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Visualize every N-th sample. Ignored if --segment is set.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (1 recommended)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id(s)')
    parser.add_argument('--segment', type=str, default=None,
                        help='If set, visualize all frames of this segment only')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    configs = load_config_module(args.config)
    val_dataset = configs.val_dataset()
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
    )
    gt_root = configs.val['gt']

    model = configs.model()
    model = load_model(model, args.checkpoint)
    print(f'[vis] Checkpoint loaded: {args.checkpoint}')
    model.cuda()
    model.eval()

    visualizer = LaneVisualizer(
        model,
        save_root=args.save_root,
        img_h=configs.input_shape[0],
        img_w=configs.input_shape[1],
        x_range=configs.x_range,
        y_range=configs.y_range,
        meter_per_pixel=configs.meter_per_pixel,
    )
    print(f'[vis] Results will be saved to: {args.save_root}')

    segment_filter = args.segment
    if segment_filter:
        max_samples = float('inf')
        interval = 1
        print(f'[vis] Segment filter: {segment_filter} (max_samples / sample_interval ignored)')
    else:
        max_samples = args.max_samples if args.max_samples > 0 else float('inf')
        interval = args.sample_interval
        print(f'[vis] Sampling every {interval}-th sample, max {max_samples} saves')

    count = 0
    global_idx = 0

    with torch.no_grad():
        for item in tqdm(val_loader, desc='Visualizing'):
            if count >= max_samples:
                break

            image, bn_name, intrinsic, road2cam, heightmap_gt = item
            bs = image.shape[0]

            # skip forward pass if no samples in this batch need visualization
            if not segment_filter:
                any_vis = any((global_idx + b) % interval == 0 for b in range(bs))
                if not any_vis:
                    global_idx += bs
                    continue

            image = image.cuda()
            intrinsic = intrinsic.cuda()
            road2cam = road2cam.cuda()

            result = model(image, intrinsic, road2cam)
            if not isinstance(result, (tuple, list)):
                result = (result,)
            pred_ = result[0]
            height_ = result[1] if len(result) > 1 else None

            if height_ is not None and height_.dim() == 3:
                height_ = height_.unsqueeze(1)

            for b_idx in range(bs):
                if count >= max_samples:
                    break

                scene = bn_name[0][b_idx]
                frame = bn_name[1][b_idx].replace('.json', '')

                if segment_filter:
                    if scene != segment_filter:
                        continue
                else:
                    if (global_idx + b_idx) % interval != 0:
                        continue

                sample_name = f'{scene}__{frame}'
                gt_path = os.path.join(gt_root, scene, bn_name[1][b_idx])
                original_img = denormalize_image(image[b_idx].cpu())

                visualizer.visualize(
                    original_img, sample_name,
                    batch_idx=b_idx,
                    pred_=pred_,
                    height_pred=height_,
                    heightmap_gt=heightmap_gt,
                    intrinsic=intrinsic,
                    extrinsic=road2cam,
                    gt_path=gt_path,
                )
                count += 1

                if count % 5 == 0:
                    print(f'[vis]  {count} saved  (scanned {global_idx + b_idx + 1} samples)')

            global_idx += bs

    print(f'\n[vis] Done! Total {count} samples saved to {args.save_root}')


if __name__ == '__main__':
    main()
