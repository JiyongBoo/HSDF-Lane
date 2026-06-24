import json
import os

import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from utils.coord_util import IPM2ego_matrix


_INV_CAMERAW2CAMERA = np.array(
    [[ 0.,  0.,  1.,  0.],
     [-1.,  0.,  0.,  0.],
     [ 0., -1.,  0.,  0.],
     [ 0.,  0.,  0.,  1.]],
    dtype=np.float64,
)


# ===========================================================================
# Pointcloud / BEV helpers
# ===========================================================================

def extract_heightmap_within_range(point_cloud_vehicle, forward_range, lateral_range):
    """Filter point cloud to the specified BEV range."""
    x = point_cloud_vehicle[:, 0]
    y = point_cloud_vehicle[:, 1]
    mask = (
        (x >= lateral_range[0]) & (x <= lateral_range[1]) &
        (y >= forward_range[0]) & (y <= forward_range[1])
    )
    return point_cloud_vehicle[mask]


def bev2ipm(bev, matrix_IPM2ego):
    """Convert BEV ego coordinates to IPM pixel coordinates.

    Args:
        bev            : (3, N) — [longitudinal, -lateral, z]
        matrix_IPM2ego : (2, 3) affine mapping IPM row/col to ego.
    Returns:
        (3, N) — [ipm_row, ipm_col, z]
    """
    ego_points = np.array([bev[0], bev[1]])
    ipm_points = (
        np.linalg.inv(matrix_IPM2ego[:, :2])
        @ (ego_points[:2] - matrix_IPM2ego[:, 2].reshape(2, 1))
    )
    return np.concatenate([ipm_points, np.array([bev[2]])], axis=0)


def generate_bev_height_map(point_cloud, resolution=(200, 48)):
    """Rasterize IPM-frame point cloud into a BEV height map.

    Args:
        point_cloud : (N, 3) — [row, col, z].
        resolution  : (H, W) output grid size.
    """
    bev_h, bev_w = resolution
    z_sum  = np.zeros((bev_h, bev_w))
    count  = np.zeros((bev_h, bev_w))
    rows   = point_cloud[:, 0].astype(int)
    cols   = point_cloud[:, 1].astype(int)
    z_vals = point_cloud[:, 2]
    valid  = (rows >= 0) & (rows < bev_h) & (cols >= 0) & (cols < bev_w)
    np.add.at(z_sum,  (rows[valid], cols[valid]), z_vals[valid])
    np.add.at(count,  (rows[valid], cols[valid]), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        height_map = np.divide(
            z_sum, count,
            out=np.zeros_like(z_sum),
            where=(count != 0),
        )
    return height_map


def generate_binary_mask(bev_height_map):
    """Return a binary mask where height != 0."""
    mask = np.zeros_like(bev_height_map)
    mask[bev_height_map != 0] = 1
    return mask


# ===========================================================================
# CenterNet heatmap helpers
# ===========================================================================

def _gaussian2d(shape, sigma=1.0):
    """2D Gaussian kernel (unnormalized, peak=1.0)."""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def _draw_gaussian(heatmap, center_y, center_x, radius, sigma):
    """Stamp a Gaussian blob onto heatmap in-place using max-pooling."""
    diameter = 2 * radius + 1
    gaussian  = _gaussian2d((diameter, diameter), sigma=sigma)
    height, width = heatmap.shape
    left   = min(center_x, radius)
    right  = min(width  - 1 - center_x, radius)
    top    = min(center_y, radius)
    bottom = min(height - 1 - center_y, radius)
    if left + right <= 0 or top + bottom <= 0:
        return
    np.maximum(
        heatmap[center_y - top:center_y + bottom + 1,
                center_x - left:center_x + right  + 1],
        gaussian[radius - top:radius + bottom + 1,
                 radius - left:radius + right  + 1],
        out=heatmap[center_y - top:center_y + bottom + 1,
                    center_x - left:center_x + right  + 1],
    )


def render_centernet_heatmap(instance_map, sigma=2.0, radius=6):
    """Build a CenterNet-style heatmap from a 1-pixel skeleton instance map.

    Args:
        instance_map : (H, W) int array — 0=background, >0=lane_id.
    Returns:
        (H, W) float32 heatmap in [0, 1].
    """
    H, W    = instance_map.shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    ys, xs  = np.where(instance_map > 0)
    for y, x in zip(ys, xs):
        _draw_gaussian(heatmap, int(y), int(x), radius, sigma)
    return heatmap


# ===========================================================================
# Training Dataset
# ===========================================================================

class Apollo_dataset_with_offset(Dataset):
    """Apollo 3D lane training dataset.

    Args:
        data_json_path   : Path to JSONL annotation file.
        dataset_base_dir : Dataset root containing images/ and map/.
        x_range          : (min, max) longitudinal range [m].
        y_range          : (min, max) lateral range [m].
        meter_per_pixel  : BEV grid resolution (m/px).
        data_trans       : Albumentations image transform.
        output_2d_shape  : 2D image GT mask size (H, W).
        input_shape      : Model input image size (H, W).
        heatmap_sigma    : Gaussian sigma for CenterNet heatmap.
        heatmap_radius   : Gaussian radius for CenterNet heatmap.

    __getitem__ returns (12 tensors):
        image, bev_gt_segment, bev_gt_instance, bev_gt_offset, bev_gt_z,
        image_gt_segment, image_gt_instance,
        intrinsic, extrinsic, road2cam, gt_heightmap, lane_heatmap_gt
    """

    def __init__(
        self,
        data_json_path,
        dataset_base_dir,
        x_range,
        y_range,
        meter_per_pixel,
        data_trans,
        output_2d_shape,
        input_shape=(600, 800),
        heatmap_sigma=2.0,
        heatmap_radius=6,
    ):
        self.x_range          = x_range
        self.y_range          = y_range
        self.meter_per_pixel  = meter_per_pixel
        self.input_shape      = input_shape
        self.dataset_base_dir = dataset_base_dir
        self.output2d_size    = output_2d_shape
        self.trans_image      = data_trans
        self.heatmap_sigma    = heatmap_sigma
        self.heatmap_radius   = heatmap_radius

        self.lane3d_thick = 1
        self.lane2d_thick = 3

        self.cnt_list = []
        with open(data_json_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.cnt_list.append(json.loads(line))
        print(f'[Apollo train] {len(self.cnt_list)} samples loaded.')

        self.ipm_h = int((self.x_range[1] - self.x_range[0]) / self.meter_per_pixel)
        self.ipm_w = int((self.y_range[1] - self.y_range[0]) / self.meter_per_pixel)

        self.matrix_IPM2ego = IPM2ego_matrix(
            ipm_center=(
                int(self.x_range[1] / self.meter_per_pixel),
                int(self.y_range[1] / self.meter_per_pixel),
            ),
            m_per_pixel=self.meter_per_pixel,
        )

    def get_camera_matrix(self, cam_pitch, cam_height):
        """Return Apollo ground-to-camera extrinsic and intrinsic matrices."""
        a = np.pi / 2 + cam_pitch
        proj_g2c = np.array(
            [[1,          0,           0,          0],
             [0,  np.cos(a),  -np.sin(a), cam_height],
             [0,  np.sin(a),   np.cos(a),          0],
             [0,          0,           0,          1]],
            dtype=np.float64,
        )
        camera_K = np.array(
            [[2015.,    0., 960.],
             [   0., 2015., 540.],
             [   0.,    0.,   1.]],
            dtype=np.float64,
        )
        return proj_g2c, camera_K

    def get_y_offset_and_z(self, res_d):
        """Convert IPM lane points to BEV offset and z maps."""

        def _calc_dist(base_pt, lane_pts, lane_z, lane_pts_set):
            cond = np.where(
                (lane_pts_set[0] == int(base_pt[0]))
                & (lane_pts_set[1] == int(base_pt[1]))
            )
            if len(cond[0]) == 0:
                return None, None
            sel_pts = lane_pts.T[cond]
            sel_z   = lane_z.T[cond]
            offset_y = np.mean(sel_pts[:, 1]) - base_pt[1]
            z        = np.mean(sel_z[:, 1])
            return offset_y, z

        res_pts     = {}
        res_pts_z   = {}
        res_pts_bin = {}
        res_pts_set = {}

        for idx, ipm_raw in res_d.items():
            ipm_arr  = np.array(ipm_raw)
            if ipm_arr.ndim != 2 or ipm_arr.shape[0] < 3:
                continue
            valid    = (ipm_arr[1] >= 0) & (ipm_arr[1] < self.ipm_h)
            ipm_filt = ipm_arr[:, valid]
            if ipm_filt.shape[1] <= 1:
                continue

            x, y, z = ipm_filt[1], ipm_filt[0], ipm_filt[2]

            x_arr, y_arr, z_arr = x, y, z
            if len(np.unique(x)) != len(x):
                sidx   = np.argsort(x)[::-1]
                xr, yr, zr = [], [], []
                prev_x = None
                for k in sidx:
                    if prev_x is not None and x[k] >= prev_x:
                        continue
                    xr.append(x[k]);  yr.append(y[k]);  zr.append(z[k])
                    prev_x = x[k]
                x_arr = np.array(xr)
                y_arr = np.array(yr)
                z_arr = np.array(zr)

            n = len(x_arr)
            if n <= 1:
                continue
            kind = 'linear' if n <= 2 else ('quadratic' if n <= 3 else 'cubic')
            fn_y = interp1d(x_arr, y_arr, kind=kind, fill_value='extrapolate')
            fn_z = interp1d(x_arr, z_arr, kind=kind)

            base     = np.linspace(x_arr.min(), x_arr.max(),
                                   max(2, int((x_arr.max() - x_arr.min()) // 0.05)))
            base_bin = np.linspace(int(x_arr.min()), int(x_arr.max()),
                                   int(int(x_arr.max()) - int(x_arr.min())) + 1)

            res_pts[idx]     = np.array([base,     fn_y(base)])
            res_pts_z[idx]   = np.array([base,     fn_z(base)])
            res_pts_bin[idx] = np.array([base_bin, fn_y(base_bin)]).astype(int)
            res_pts_set[idx] = np.array([base,     fn_y(base)]).astype(int)

        offset_map = np.zeros((self.ipm_h, self.ipm_w))
        z_map      = np.zeros((self.ipm_h, self.ipm_w))
        ipm_image  = np.zeros((self.ipm_h, self.ipm_w))

        for idx, pts_bin in res_pts_bin.items():
            for pt in pts_bin.T:
                row, col = int(pt[0]), int(pt[1])
                if not (0 < row < self.ipm_h and 0 < col < self.ipm_w):
                    continue
                ipm_image[row, col] = idx
                offset_y, z = _calc_dist(
                    np.array([row, col]),
                    res_pts[idx], res_pts_z[idx], res_pts_set[idx],
                )
                if offset_y is None:
                    ipm_image[row, col] = 0
                    continue
                offset_map[row, col] = max(0.0, min(1.0, offset_y))
                z_map[row, col]      = z

        return ipm_image, offset_map, z_map

    def get_seg_offset(self, idx):
        """Load and preprocess a single training sample."""
        info_dict = self.cnt_list[idx]
        name_list = info_dict['raw_file'].split('/')

        image_path = os.path.join(
            self.dataset_base_dir, 'images', name_list[-2], name_list[-1]
        )
        map_path = os.path.join(
            self.dataset_base_dir, 'map',
            name_list[-2],
            name_list[-1].replace('jpg', 'npy'),
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cam_height = info_dict['cam_height']
        cam_pitch  = info_dict['cam_pitch']
        proj_g2c, camera_k = self.get_camera_matrix(cam_pitch, cam_height)

        _raw = np.load(map_path)
        if _raw.ndim == 3 and _raw.shape[0] == 2:
            bev_height_map_mask = _raw.astype(np.float32)
        else:
            roi_pc = extract_heightmap_within_range(_raw, self.x_range, self.y_range)
            if roi_pc.shape[0] > 0:
                roi_bev = np.array([roi_pc[:, 1], -roi_pc[:, 0], roi_pc[:, 2]])
                ipm_roi = bev2ipm(roi_bev, self.matrix_IPM2ego)
                bev_heightmap = generate_bev_height_map(
                    ipm_roi.T,
                    resolution=(self.ipm_h, self.ipm_w),
                )
            else:
                bev_heightmap = np.zeros((self.ipm_h, self.ipm_w))
            bev_height_map_mask = np.stack(
                (bev_heightmap, generate_binary_mask(bev_heightmap)), axis=0
            )

        lane_grounds = info_dict['laneLines']
        image_gt     = np.zeros(image.shape[:2], dtype=np.uint8)
        res_points_d = {}

        for lane_idx, lane_raw in enumerate(lane_grounds):
            vis         = np.array(info_dict['laneLines_visibility'][lane_idx])
            lane_ground = np.array(lane_raw)
            assert vis.shape[0] == lane_ground.shape[0], \
                f"visibility mismatch at idx={idx} lane={lane_idx}"
            lane_ground = lane_ground[vis > 0.5]
            if len(lane_ground) == 0:
                continue

            lg_h = np.concatenate(
                [lane_ground, np.ones([lane_ground.shape[0], 1])], axis=1
            ).T

            lane_cam = proj_g2c @ lg_h
            lane_img = camera_k @ lane_cam[:3]
            lane_img = lane_img / lane_img[2]
            lane_uv  = lane_img[:2].T
            cv2.polylines(image_gt, [lane_uv.astype(int)], False, lane_idx + 1, 3)

            x_lon = lg_h[1]
            y_lat = lg_h[0]
            ground_pts = np.array([x_lon, -y_lat])
            ipm_pts    = (
                np.linalg.inv(self.matrix_IPM2ego[:, :2])
                @ (ground_pts - self.matrix_IPM2ego[:, 2].reshape(2, 1))
            )
            ipm_pts_sw    = np.zeros((3, ipm_pts.shape[1]))
            ipm_pts_sw[0] = ipm_pts[1]
            ipm_pts_sw[1] = ipm_pts[0]
            ipm_pts_sw[2] = lg_h[2]
            res_points_d[lane_idx + 1] = ipm_pts_sw

        bev_gt, offset_y_map, z_map = self.get_y_offset_and_z(res_points_d)
        return (
            image, image_gt, bev_gt, offset_y_map, z_map,
            proj_g2c, camera_k, bev_height_map_mask,
        )

    def __getitem__(self, idx):
        (
            image, image_gt, bev_gt, offset_y_map, z_map,
            cam_extrinsics, cam_intrinsic, bev_height_map_mask,
        ) = self.get_seg_offset(idx)

        image_h, image_w = image.shape[:2]
        in_h, in_w       = self.input_shape

        cam_intrinsic = cam_intrinsic.copy()
        cam_intrinsic[0] *= in_w / image_w
        cam_intrinsic[1] *= in_h / image_h

        road2cam_np = _INV_CAMERAW2CAMERA @ cam_extrinsics

        transformed = self.trans_image(image=image)
        image       = transformed['image']

        image_gt = cv2.resize(
            image_gt,
            (self.output2d_size[1], self.output2d_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        intrinsic = torch.tensor(cam_intrinsic)
        extrinsic = torch.tensor(cam_extrinsics)
        road2cam  = torch.tensor(road2cam_np)

        image_gt_instance = torch.tensor(image_gt).unsqueeze(0)
        image_gt_segment  = torch.clone(image_gt_instance)
        image_gt_segment[image_gt_segment > 0] = 1

        bev_gt_instance = torch.tensor(bev_gt).unsqueeze(0)
        bev_gt_offset   = torch.tensor(offset_y_map).unsqueeze(0)
        bev_gt_z        = torch.tensor(z_map).unsqueeze(0)
        bev_gt_segment  = torch.clone(bev_gt_instance)
        bev_gt_segment[bev_gt_segment > 0] = 1

        gt_heightmap = torch.tensor(bev_height_map_mask)

        heatmap_np = render_centernet_heatmap(
            bev_gt.astype(np.int32),
            sigma=self.heatmap_sigma,
            radius=self.heatmap_radius,
        )
        lane_heatmap_gt = torch.from_numpy(heatmap_np).unsqueeze(0)

        return (
            image,
            bev_gt_segment.float(),
            bev_gt_instance.float(),
            bev_gt_offset.float(),
            bev_gt_z.float(),
            image_gt_segment.float(),
            image_gt_instance.float(),
            intrinsic.float(),
            extrinsic.float(),
            road2cam.float(),
            gt_heightmap.float(),
            lane_heatmap_gt.float(),
        )

    def __len__(self):
        return len(self.cnt_list)


# ===========================================================================
# Validation Dataset
# ===========================================================================

class Apollo_dataset_with_offset_val(Dataset):
    """Apollo 3D lane validation dataset.

    __getitem__ returns:
        (image, [folder, filename], intrinsic, road2cam, gt_heightmap)

    Args:
        data_json_path   : Path to JSONL annotation file.
        dataset_base_dir : Dataset root containing images/ and map/.
        data_trans       : Image transform.
        x_range          : Longitudinal range [m].
        y_range          : Lateral range [m].
        meter_per_pixel  : BEV resolution.
        input_shape      : Model input image size (H, W).
    """

    def __init__(
        self,
        data_json_path,
        dataset_base_dir,
        data_trans,
        x_range=(3., 103.),
        y_range=(-12., 12.),
        meter_per_pixel=0.5,
        input_shape=(600, 800),
    ):
        self.dataset_base_dir = dataset_base_dir
        self.trans_image      = data_trans
        self.input_shape      = input_shape
        self.x_range          = x_range
        self.y_range          = y_range
        self.meter_per_pixel  = meter_per_pixel

        self.ipm_h = int((x_range[1] - x_range[0]) / meter_per_pixel)
        self.ipm_w = int((y_range[1] - y_range[0]) / meter_per_pixel)

        self.cnt_list = []
        with open(data_json_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.cnt_list.append(json.loads(line))

        self._name_to_idx = {
            (info['raw_file'].split('/')[-2],
             info['raw_file'].split('/')[-1]): i
            for i, info in enumerate(self.cnt_list)
        }
        print(f'[Apollo val] {len(self.cnt_list)} samples loaded.')

    def get_camera_matrix(self, cam_pitch, cam_height):
        """Return Apollo ground-to-camera extrinsic and intrinsic matrices."""
        a = np.pi / 2 + cam_pitch
        proj_g2c = np.array(
            [[1,          0,           0,          0],
             [0,  np.cos(a),  -np.sin(a), cam_height],
             [0,  np.sin(a),   np.cos(a),          0],
             [0,          0,           0,          1]],
            dtype=np.float64,
        )
        camera_K = np.array(
            [[2015.,    0., 960.],
             [   0., 2015., 540.],
             [   0.,    0.,   1.]],
            dtype=np.float64,
        )
        return proj_g2c, camera_K

    def __getitem__(self, idx):
        info_dict = self.cnt_list[idx]
        name_list = info_dict['raw_file'].split('/')

        image_path = os.path.join(
            self.dataset_base_dir, 'images', name_list[-2], name_list[-1]
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_h, image_w = image.shape[:2]
        in_h, in_w       = self.input_shape

        cam_height = info_dict['cam_height']
        cam_pitch  = info_dict['cam_pitch']
        proj_g2c, camera_k = self.get_camera_matrix(cam_pitch, cam_height)

        camera_k = camera_k.copy()
        camera_k[0] *= in_w / image_w
        camera_k[1] *= in_h / image_h

        road2cam = _INV_CAMERAW2CAMERA @ proj_g2c

        transformed = self.trans_image(image=image)
        image       = transformed['image']

        map_path = os.path.join(
            self.dataset_base_dir, 'map',
            name_list[-2],
            name_list[-1].replace('jpg', 'npy'),
        )
        try:
            _raw = np.load(map_path)
            if _raw.ndim == 3 and _raw.shape[0] == 2:
                gt_heightmap = torch.tensor(_raw.astype(np.float32))
            else:
                gt_heightmap = torch.zeros(2, self.ipm_h, self.ipm_w)
        except Exception:
            gt_heightmap = torch.zeros(2, self.ipm_h, self.ipm_w)

        return (
            image,
            [name_list[-2], name_list[-1]],
            torch.tensor(camera_k).float(),
            torch.tensor(road2cam).float(),
            gt_heightmap.float(),
        )

    def __len__(self):
        return len(self.cnt_list)
