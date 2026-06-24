"""
val_apollo.py  —  Single-checkpoint validation for Apollo dataset
=================================================================
Runs validation on a given config + checkpoint and prints results
to console and saves them to a txt file.

Usage:
    python tools/val_apollo.py \\
        --config ./tools/hsdflane_apollo_config.py \\
        --checkpoint ./hsdflane_apollo_standard/best.pth

    python tools/val_apollo.py \\
        --config ./tools/hsdflane_apollo_config.py \\
        --checkpoint ./hsdflane_apollo_standard/best.pth \\
        --device cuda:1 --batch_size 8 --output ./results.txt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if 'tools' in sys.modules:
    del sys.modules['tools']

import re
import time
import argparse
import warnings
from datetime import datetime

import torch
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
from torch.utils.data import DataLoader

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

from utils.config_util import load_config_module
from models.util.cluster import embedding_post
from models.util.post_process import bev_instance2points_with_offset_z
from models.util.load_model import load_model
from utils.util_val.val_offical import LaneEval

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


# ===========================================================================
# Load Apollo GT
# ===========================================================================

def get_gt_lanes_apollo(info_dict):
    lanes        = info_dict['laneLines']
    visibilities = info_dict['laneLines_visibility']
    frame_lanes  = []
    for lane_raw, vis_raw in zip(lanes, visibilities):
        pts = np.array(lane_raw)
        vis = np.array(vis_raw)
        pts = pts[vis > 0.5]
        if len(pts) < 2:
            continue
        dist = np.sqrt(
            (pts[-1, 1] - pts[0, 1]) ** 2 +
            (pts[-1, 0] - pts[0, 0]) ** 2
        )
        if dist < 3.0:
            continue
        lane_persformer = np.stack([-pts[:, 0], pts[:, 1], pts[:, 2]], axis=1)
        frame_lanes.append(lane_persformer.tolist())
    return frame_lanes


# ===========================================================================
# AP calculation helpers
# ===========================================================================

def compute_lane_pr(cached_frames, conf_thresh, x_range, meter_per_pixel,
                    post_emb_margin=6.0, post_min_cluster_size=15):
    lane_eval = LaneEval()
    for fr in cached_frames:
        ms       = fr['seg']
        me       = fr['emb']
        moffset  = fr['offset_y']
        z        = fr['height']
        gt_lanes = fr['gt_lanes']

        canvas, _ = embedding_post(
            (ms, me),
            conf=conf_thresh,
            emb_margin=post_emb_margin,
            min_cluster_size=post_min_cluster_size,
            canvas_color=False,
        )
        lines = bev_instance2points_with_offset_z(
            canvas,
            max_x=x_range[1],
            meter_per_pixal=(meter_per_pixel, meter_per_pixel),
            offset_y=moffset[0][0],
            Z=z[0][0],
        )
        frame_lanes_pred = []
        for lane in lines:
            pred_in_persformer = np.array([lane[1], lane[0], lane[2]])
            frame_lanes_pred.append(pred_in_persformer.T.tolist())
        lane_eval.bench_all(frame_lanes_pred, gt_lanes)

    r_lane   = np.sum(lane_eval.r_list)
    p_lane   = np.sum(lane_eval.p_list)
    cnt_gt   = np.sum(lane_eval.cnt_gt_list)
    cnt_pred = np.sum(lane_eval.cnt_pred_list)
    recall    = r_lane   / (cnt_gt   + 1e-6)
    precision = p_lane   / (cnt_pred + 1e-6)
    return precision, recall


def compute_ap_interp1d(precisions, recalls):
    R_arr = np.array([1.] + list(recalls) + [0.])
    P_arr = np.array([0.] + list(precisions) + [1.])
    try:
        f = interp1d(R_arr, P_arr)
        r_range = np.clip(np.linspace(0.05, 0.95, 19), R_arr.min(), R_arr.max())
        ap = float(np.mean(f(r_range)))
    except Exception as e:
        print(f"[warn] interp1d AP 계산 실패 ({e}), trapz AUC로 대체")
        order = np.argsort(R_arr)
        ap = float(np.trapz(P_arr[order], R_arr[order]))
    return ap


def _collect_height_metrics(abs_errors):
    if abs_errors:
        all_abs_err = np.concatenate(abs_errors)
        return {
            'height_mae':     float(np.mean(all_abs_err)),
            'height_rmse':    float(np.sqrt(np.mean(all_abs_err ** 2))),
            'height_acc_005': float(np.mean(all_abs_err < 0.05)),
            'height_acc_010': float(np.mean(all_abs_err < 0.10)),
            'height_acc_020': float(np.mean(all_abs_err < 0.20)),
        }
    return {
        'height_mae': 0.0, 'height_rmse': 0.0,
        'height_acc_005': 0.0, 'height_acc_010': 0.0, 'height_acc_020': 0.0,
    }


def extract_epoch(checkpoint_path):
    match = re.search(r'ep(\d+)', os.path.basename(checkpoint_path))
    if match:
        return int(match.group(1))
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        return ckpt.get('epoch', -1)
    except:
        return -1


def _load_model_fallback(model, ckpt_path):
    print("  [Fallback] Using custom weight loading with key remapping...")
    ckpt_data = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt_data['model_state']

    cleaned = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('model.'):
            key = key[len('model.'):]
        cleaned[key] = v

    loss_keys = [k for k in cleaned if k.startswith(('_gauss_kernel', 'bce.', 'loss.'))]
    for k in loss_keys:
        del cleaned[k]

    remap = {}
    drop = []
    for k in list(cleaned.keys()):
        if '.coord_embed.3.' in k:
            key = k.replace('.coord_embed.3.', '.coord_embed.2.')
            remap[k] = key
        elif '.coord_embed.1.' in k and ('running_mean' in k or 'running_var' in k
                                          or 'num_batches_tracked' in k):
            drop.append(k)
        elif '.coord_embed.1.weight' in k or '.coord_embed.1.bias' in k:
            prefix = k.rsplit('.coord_embed.1.', 1)[0]
            bn_check = f"{prefix}.coord_embed.1.running_mean"
            if bn_check in cleaned:
                drop.append(k)
    for k in drop:
        del cleaned[k]
    for old_k, new_k in remap.items():
        cleaned[new_k] = cleaned.pop(old_k)

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"  [Warning] load_state_dict strict=False")
        if missing:
            print(f"    Missing  : {missing}")
        if unexpected:
            print(f"    Unexpected: {unexpected}")
    del ckpt_data, state_dict, cleaned
    return model


def load_model_and_epoch(model, ckpt_path):
    try:
        model = load_model(model, ckpt_path)
        print("  Weight loading: standard (load_model)")
    except Exception as e:
        print(f"  [Warning] load_model() failed: {e}")
        model = _load_model_fallback(model, ckpt_path)
        print("  Weight loading: fallback (key remap)")
    epoch = extract_epoch(ckpt_path)
    return model, epoch


@torch.no_grad()
def measure_flops(model, val_loader, device):
    if not HAS_THOP:
        print("  [Warning] thop not installed. Run: pip install thop")
        total_params = sum(p.numel() for p in model.parameters())
        return None, total_params

    sample_batch = next(iter(val_loader))
    image_sample     = sample_batch[0][:1].to(device)
    intrinsic_sample = sample_batch[2][:1].to(device)
    road2cam_sample  = sample_batch[3][:1].to(device)

    flops, params = thop_profile(
        model,
        inputs=(image_sample, intrinsic_sample, road2cam_sample),
        verbose=False,
    )
    return flops, params


@torch.no_grad()
def measure_fps(model, val_loader, device, warmup_batches=10, measure_batches=50,
                track_peak_mem=False):
    model.eval()
    loader_iter = iter(val_loader)

    if track_peak_mem and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(val_loader)
            batch = next(loader_iter)
        image    = batch[0].to(device, non_blocking=True)
        intrinsic = batch[2].to(device, non_blocking=True)
        road2cam  = batch[3].to(device, non_blocking=True)
        _ = model(image, intrinsic, road2cam)
    torch.cuda.synchronize()

    total_frames = 0
    start = time.time()
    for _ in range(measure_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(val_loader)
            batch = next(loader_iter)
        image    = batch[0].to(device, non_blocking=True)
        intrinsic = batch[2].to(device, non_blocking=True)
        road2cam  = batch[3].to(device, non_blocking=True)
        torch.cuda.synchronize()
        _ = model(image, intrinsic, road2cam)
        torch.cuda.synchronize()
        total_frames += image.shape[0]
    elapsed = time.time() - start
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    if track_peak_mem:
        peak_alloc = peak_reserved = None
        if device.type == 'cuda':
            torch.cuda.synchronize()
            peak_alloc    = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        return fps, total_frames, elapsed, peak_alloc, peak_reserved
    return fps, total_frames, elapsed



@torch.no_grad()
def validate(model, val_loader, val_dataset, configs, device, epoch=-1):
    model.eval()
    x_range          = configs.x_range
    meter_per_pixel  = configs.meter_per_pixel
    post_conf            = getattr(configs, 'post_conf', -1.7)
    post_emb_margin      = getattr(configs, 'post_emb_margin', 6.0)
    post_min_cluster_size = getattr(configs, 'post_min_cluster_size', 15)

    lane_eval       = LaneEval()
    height_abs_errors = []
    cached_frames   = []
    total_infer_time = 0.0
    total_frames    = 0

    for batch in tqdm(val_loader, desc=f"Validation (epoch {epoch})"):
        image, bn_name, intrinsic, road2cam, heightmap_gt = batch
        image     = image.to(device, non_blocking=True)
        intrinsic = intrinsic.to(device, non_blocking=True)
        road2cam  = road2cam.to(device, non_blocking=True)

        torch.cuda.synchronize()
        infer_start = time.time()
        model_out = model(image, intrinsic, road2cam)
        torch.cuda.synchronize()
        total_infer_time += time.time() - infer_start
        total_frames += image.shape[0]

        pred_   = model_out[0]
        height_ = model_out[1]

        seg       = pred_[0].cpu().numpy()
        embedding = pred_[1].cpu().numpy()
        offset_y  = torch.sigmoid(pred_[2]).cpu().numpy()

        if isinstance(height_, list):
            height_ = height_[-1]
        if height_.dim() == 3:
            height_ = height_.unsqueeze(1)
        height = height_.cpu().numpy()

        gt_height_np = heightmap_gt[:, 0].numpy()
        gt_mask_np   = heightmap_gt[:, 1].numpy()

        folders   = bn_name[0]
        filenames = bn_name[1]
        batch_size = seg.shape[0]

        for idx in range(batch_size):
            valid = gt_mask_np[idx] > 0.5
            if valid.sum() > 0:
                abs_err = np.abs(height[idx, 0][valid] - gt_height_np[idx][valid])
                height_abs_errors.append(abs_err)

            ms      = seg[idx:idx + 1]
            me      = embedding[idx:idx + 1]
            moffset = offset_y[idx:idx + 1]
            z       = height[idx:idx + 1]

            folder   = folders[idx]
            filename = filenames[idx]
            list_idx = val_dataset._name_to_idx.get((folder, filename), None)
            if list_idx is None:
                print(f"[warn] GT not found: {folder}/{filename}")
                gt_lanes = []
            else:
                gt_lanes = get_gt_lanes_apollo(val_dataset.cnt_list[list_idx])

            cached_frames.append({
                'seg':      ms,
                'emb':      me,
                'offset_y': moffset,
                'height':   z,
                'gt_lanes': gt_lanes,
            })

            canvas, _ = embedding_post(
                (ms, me),
                conf=post_conf,
                emb_margin=post_emb_margin,
                min_cluster_size=post_min_cluster_size,
                canvas_color=False,
            )
            lines = bev_instance2points_with_offset_z(
                canvas, max_x=x_range[1],
                meter_per_pixal=(meter_per_pixel, meter_per_pixel),
                offset_y=moffset[0][0], Z=z[0][0],
            )
            frame_lanes_pred = []
            for lane in lines:
                # Apollo coordination
                pred_in_persformer = np.array([lane[1], lane[0], lane[2]])
                frame_lanes_pred.append(pred_in_persformer.T.tolist())
            lane_eval.bench_all(frame_lanes_pred, gt_lanes)

    results = lane_eval.show()
    results.update(_collect_height_metrics(height_abs_errors))
    results['fps'] = total_frames / total_infer_time if total_infer_time > 0 else 0.0
    return results, cached_frames


def run_ap_sweep(cached_frames, configs):
    x_range           = configs.x_range
    meter_per_pixel   = configs.meter_per_pixel
    post_conf         = getattr(configs, 'post_conf', -1.7)
    post_emb_margin   = getattr(configs, 'post_emb_margin', 6.0)
    post_min_cluster_size = getattr(configs, 'post_min_cluster_size', 15)

    ap_conf_range = getattr(configs, 'ap_conf_range',
                             [-4.0, -3.5, -3.0, -2.5, -2.0, post_conf,
                              -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    ap_conf_range = sorted(set(ap_conf_range))

    all_precisions, all_recalls = [], []
    for thr in tqdm(ap_conf_range, desc="AP sweep"):
        p, r = compute_lane_pr(cached_frames, thr, x_range, meter_per_pixel,
                                post_emb_margin, post_min_cluster_size)
        all_precisions.append(p)
        all_recalls.append(r)

    ap = compute_ap_interp1d(all_precisions, all_recalls)
    f1_list    = [2 * p * r / (p + r + 1e-6) for p, r in zip(all_precisions, all_recalls)]
    max_f1_idx = int(np.argmax(f1_list))
    return ap, ap_conf_range, all_precisions, all_recalls, f1_list, max_f1_idx




def _format_lane_height_block(res, label):
    lines = []
    lines.append(f"── {label} / Lane Detection ──")
    lines.append(f"  F1         : {res.get('f1_score', 0):.4f}")
    lines.append(f"  Precision  : {res.get('precision', 0):.4f}")
    lines.append(f"  Recall     : {res.get('recall', 0):.4f}")
    lines.append(f"  x_err_close: {res.get('x_error_close', 0):.4f}")
    lines.append(f"  x_err_far  : {res.get('x_error_far', 0):.4f}")
    lines.append(f"  z_err_close: {res.get('z_error_close', 0):.4f}")
    lines.append(f"  z_err_far  : {res.get('z_error_far', 0):.4f}")
    lines.append("")
    lines.append(f"── {label} / Height Estimation ──")
    lines.append(f"  MAE        : {res.get('height_mae', 0):.4f}")
    lines.append(f"  RMSE       : {res.get('height_rmse', 0):.4f}")
    lines.append(f"  Acc@0.05   : {res.get('height_acc_005', 0):.4f}")
    lines.append(f"  Acc@0.10   : {res.get('height_acc_010', 0):.4f}")
    lines.append(f"  Acc@0.20   : {res.get('height_acc_020', 0):.4f}")
    return lines


def format_results(results, ap_data, config_path, ckpt_path, epoch, configs):
    post_conf             = getattr(configs, 'post_conf', -1.7)
    post_emb_margin       = getattr(configs, 'post_emb_margin', 6.0)
    post_min_cluster_size = getattr(configs, 'post_min_cluster_size', 15)
    split_type            = getattr(configs, 'SPLIT_TYPE', 'unknown')

    ap, ap_conf_range, precisions, recalls, f1_list, max_f1_idx = ap_data

    lines = []
    lines.append("=" * 60)
    lines.append("  Apollo Validation Results")
    lines.append(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Config     : {config_path}")
    lines.append(f"  Checkpoint : {ckpt_path}")
    lines.append(f"  Epoch      : {epoch}")
    lines.append(f"  Split      : {split_type}")
    lines.append(f"  Post Conf  : {post_conf:g}")
    lines.append(f"  Post Emb   : {post_emb_margin:g}")
    lines.append(f"  Post Min   : {post_min_cluster_size:d}")
    lines.append("=" * 60)
    lines.append("")

    lines.extend(_format_lane_height_block(results, "Overall"))
    lines.append("")

    lines.append("── AP Metrics (eval_3D_lane interp1d, 19-point) ──")
    lines.append(f"  AP             : {ap:.4f}")
    lines.append(f"  Max F1         : {f1_list[max_f1_idx]:.4f}  "
                 f"(at conf={ap_conf_range[max_f1_idx]:+.2f})")
    lines.append("")

    lines.append("── PR Curve  (conf low → high) ──")
    lines.append(f"  {'conf':>8}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    for i, (thr, p, r, f1) in enumerate(zip(ap_conf_range, precisions, recalls, f1_list)):
        marker = "  ← max F1" if i == max_f1_idx else ""
        lines.append(f"  {thr:+8.2f}  {p:10.4f}  {r:10.4f}  {f1:10.4f}{marker}")
    lines.append("")

    lines.append("── Inference Speed ──")
    lines.append(f"  FPS        : {results.get('fps', 0):.2f}")
    flops  = results.get('flops', None)
    params = results.get('params', None)
    if flops is not None:
        lines.append(f"  FLOPs      : {flops / 1e9:.2f} G")
    if params is not None:
        lines.append(f"  Params     : {params / 1e6:.2f} M")
    lines.append("=" * 60)
    return "\n".join(lines)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Single-checkpoint validation for Apollo dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/val_apollo.py \\
      --config ./tools/hsdflane_apollo_config.py \\
      --checkpoint ./hsdflane_apollo_standard/best.pth

  python tools/val_apollo.py \\
      --config ./tools/hsdflane_apollo_config.py \\
      --checkpoint ./hsdflane_apollo_standard/best.pth \\
      --device cuda:1 --batch_size 8 --output ./results.txt
        """)
    parser.add_argument('--config', type=str, required=True,
        help='Path to config .py file')
    parser.add_argument('--checkpoint', type=str, required=True,
        help='Path to checkpoint .pth file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output', type=str, default=None,
        help='Output txt path (default: {ckpt_stem}_results.txt next to checkpoint)')
    parser.add_argument('--speed_only', action='store_true',
        help='Only measure FPS and FLOPs, skip full validation')
    parser.add_argument('--warmup_batches', type=int, default=100,
        help='Warmup batches for FPS measurement')
    parser.add_argument('--measure_batches', type=int, default=500,
        help='Batches used to measure FPS')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.output is None:
        ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = os.path.join(
            os.path.dirname(args.checkpoint) or '.', f'{ckpt_stem}_results.txt')

    configs = load_config_module(args.config)
    post_conf             = getattr(configs, 'post_conf', -1.7)
    post_emb_margin       = getattr(configs, 'post_emb_margin', 6.0)
    post_min_cluster_size = getattr(configs, 'post_min_cluster_size', 15)

    print(f"Config     : {args.config}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Split      : {getattr(configs, 'SPLIT_TYPE', 'unknown')}")
    print(f"Output     : {args.output}")
    print(f"Post Conf  : {post_conf:g}  |  "
          f"Post Emb: {post_emb_margin:g}  |  "
          f"Post Min: {post_min_cluster_size:d}")
    print()

    print("Loading val dataset...")
    val_dataset = configs.val_dataset()
    val_loader  = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print("Creating model...")
    model = configs.val_model() if hasattr(configs, 'val_model') else configs.model()
    model, epoch = load_model_and_epoch(model, args.checkpoint)
    model.to(device)
    model.eval()
    print(f"Loaded epoch {epoch}\n")

    print("Measuring FLOPs...")
    flops, params = measure_flops(model, val_loader, device)
    if flops is not None:
        print(f"  FLOPs  : {flops / 1e9:.2f} G")
    print(f"  Params : {params / 1e6:.2f} M\n")

    if args.speed_only:
        print(f"Measuring FPS "
              f"(warmup={args.warmup_batches}, measure={args.measure_batches} batches)...")
        fps, total_frames, elapsed, peak_alloc, peak_reserved = measure_fps(
            model, val_loader, device,
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
            track_peak_mem=True,
        )
        print(f"\n{'=' * 60}")
        print(f"  Speed-Only Results")
        print(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Config     : {args.config}")
        print(f"  Checkpoint : {args.checkpoint}")
        print(f"  Epoch      : {epoch}")
        print(f"{'=' * 60}")
        print(f"  FPS        : {fps:.2f}  ({total_frames} frames / {elapsed:.2f}s)")
        if peak_alloc is not None:
            print(f"  Peak Mem   : {peak_alloc / (1024 ** 2):.2f} MB (alloc), "
                  f"{peak_reserved / (1024 ** 2):.2f} MB (reserved)")
        else:
            print("  Peak Mem   : N/A")
        if flops is not None:
            print(f"  FLOPs      : {flops / 1e9:.2f} G")
        print(f"  Params     : {params / 1e6:.2f} M")
        print(f"{'=' * 60}")

        peak_mem_line = (
            f"Peak_Mem: {peak_alloc / (1024 ** 2):.2f} MB (alloc), "
            f"{peak_reserved / (1024 ** 2):.2f} MB (reserved)"
            if peak_alloc is not None else "Peak_Mem: N/A"
        )
        speed_lines = [
            f"FPS: {fps:.2f}",
            peak_mem_line,
            f"FLOPs: {flops / 1e9:.2f} G" if flops is not None else "FLOPs: N/A (install thop)",
            f"Params: {params / 1e6:.2f} M",
            f"Frames: {total_frames}, Time: {elapsed:.2f}s",
        ]
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write("\n".join(speed_lines) + "\n")
        print(f"\nResults saved to {args.output}")
        return

    print("Running validation...")
    results, cached_frames = validate(model, val_loader, val_dataset, configs, device, epoch)
    results['flops']  = flops
    results['params'] = params

    print(f"\nRunning AP sweep ({len(cached_frames)} frames cached)...")
    ap_data = run_ap_sweep(cached_frames, configs)

    report = format_results(results, ap_data, args.config, args.checkpoint, epoch, configs)
    print("\n" + report)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(report + "\n")
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
