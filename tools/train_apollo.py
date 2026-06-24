"""
train_apollo.py  —  Training script for Apollo dataset.
==============================================================
Runs training on a given config and saves the model to a checkpoint file.

Usage:
    python tools/train_apollo.py \\
        --config ./tools/hsdflane_config.py 
    
    # To resume training from a checkpoint:
    python tools/train_apollo.py \\
        --config ./tools/hsdflane_config.py \\
        --checkpoint ./train_results/ep.pth  
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from utils.config_util import load_config_module
from models.util.cluster import embedding_post
from models.util.post_process import bev_instance2points_with_offset_z
from utils.util_val.val_offical import LaneEval
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import warnings
import argparse
import wandb

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
torch.backends.cudnn.benchmark = True


def is_main_process(rank):
    return rank == 0


def unpack_train_outputs(outputs):
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 5:
        raise ValueError("Model forward(train=True) must return at least 5 values.")

    pred = outputs[0]
    loss_total_bev = outputs[1]
    loss_offset = outputs[2]
    loss_total_2d = outputs[3]
    loss_height = outputs[4]
    loss_details = outputs[5] if len(outputs) > 5 and isinstance(outputs[5], dict) else {}
    return pred, loss_total_bev, loss_offset, loss_total_2d, loss_height, loss_details


def apply_linear_warmup(optimizer, base_lrs, global_step, warmup_steps):
    if warmup_steps <= 0 or global_step >= warmup_steps:
        return

    warmup_ratio = float(global_step + 1) / float(warmup_steps)
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group['lr'] = base_lr * warmup_ratio


def get_gt_lanes_apollo(info_dict):
    """Load GT lanes from Apollo JSONL info_dict in persformer format.

    Apollo ego frame: [x_lat, y_lon, z]  (x_lat positive = right)
    Persformer format: [-x_lat, y_lon, z] (col0 = left-positive)

    This matches the final coordinate system of the OpenLane matrix_lane2persformer
    transform. Note: OpenLane prediction uses pred=[-lane[1], lane[0], lane[2]],
    while Apollo uses pred=[lane[1], lane[0], lane[2]] since lane[1] already
    carries the sign flip from bev_instance2points_with_offset_z.
    """
    lanes = info_dict['laneLines']
    visibilities = info_dict['laneLines_visibility']

    frame_lanes = []
    for lane_raw, vis_raw in zip(lanes, visibilities):
        pts = np.array(lane_raw)    # (N, 3): [x_lat, y_lon, z]
        vis = np.array(vis_raw)     # (N,)
        pts = pts[vis > 0.5]
        if len(pts) < 2:
            continue
        dist_val = np.sqrt(
            (pts[-1, 1] - pts[0, 1]) ** 2 +
            (pts[-1, 0] - pts[0, 0]) ** 2
        )
        if dist_val < 3.0:
            continue
        lane_persformer = np.stack([-pts[:, 0], pts[:, 1], pts[:, 2]], axis=1)
        frame_lanes.append(lane_persformer.tolist())

    return frame_lanes



def train_epoch(rank, model, dataloader, optimizer, scaler, epoch,
                use_amp=True, global_step=0, warmup_steps=0, base_lrs=None):
    model.train()
    inner = model.module if hasattr(model, 'module') else model
    if hasattr(inner, 'set_epoch'):
        inner.set_epoch(epoch)
    if base_lrs is None:
        base_lrs = [group['lr'] for group in optimizer.param_groups]

    total_loss_bev = torch.tensor(0.0, device=rank)
    total_loss_offset = torch.tensor(0.0, device=rank)
    total_loss_2d = torch.tensor(0.0, device=rank)
    total_loss_height = torch.tensor(0.0, device=rank)
    total_loss_detail = {}
    total_grad_norm = 0.0
    num_batches = 0

    if is_main_process(rank):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader

    for idx, batch in enumerate(pbar):
        # Global-step linear warmup (first warmup_steps only)
        apply_linear_warmup(optimizer, base_lrs, global_step, warmup_steps)

        # Apollo batch: no cam_w_extrinsics / vehicle_pose fields
        (input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data,
         image_gt_segment, image_gt_instance, intrinsic, extrinsic,
         road2cam, heightmap, *extra) = batch
        lane_heatmap_gt = extra[0] if extra else None

        input_data = input_data.to(rank, non_blocking=True)
        gt_seg_data = gt_seg_data.to(rank, non_blocking=True)
        gt_emb_data = gt_emb_data.to(rank, non_blocking=True)
        offset_y_data = offset_y_data.to(rank, non_blocking=True)
        z_data = z_data.to(rank, non_blocking=True)
        image_gt_segment = image_gt_segment.to(rank, non_blocking=True)
        image_gt_instance = image_gt_instance.to(rank, non_blocking=True)
        intrinsic = intrinsic.to(rank, non_blocking=True)
        road2cam = road2cam.to(rank, non_blocking=True)
        heightmap = heightmap.to(rank, non_blocking=True)
        if lane_heatmap_gt is not None:
            lane_heatmap_gt = lane_heatmap_gt.to(rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                outputs = model(
                    input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data,
                    image_gt_segment, image_gt_instance, heightmap,
                    intrinsic=intrinsic, road2cam=road2cam, train=True,
                    lane_heatmap_gt=lane_heatmap_gt,
                )
                prediction, loss_total_bev, loss_offset, loss_total_2d, loss_height, loss_details = unpack_train_outputs(outputs)
                loss_total = loss_total_bev + loss_offset + loss_total_2d + loss_height

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data,
                image_gt_segment, image_gt_instance, heightmap,
                intrinsic=intrinsic, road2cam=road2cam, train=True,
                lane_heatmap_gt=lane_heatmap_gt,
            )
            prediction, loss_total_bev, loss_offset, loss_total_2d, loss_height, loss_details = unpack_train_outputs(outputs)
            loss_total = loss_total_bev + loss_offset + loss_total_2d + loss_height
            loss_total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)
            optimizer.step()

        total_loss_bev += loss_total_bev.detach()
        total_loss_offset += loss_offset.detach()
        total_loss_2d += loss_total_2d.detach()
        total_loss_height += loss_height.detach()
        for key, value in loss_details.items():
            if torch.is_tensor(value):
                v = value.detach()
            else:
                v = torch.tensor(float(value), device=rank)
            if key not in total_loss_detail:
                total_loss_detail[key] = torch.tensor(0.0, device=rank)
            total_loss_detail[key] += v

        total_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        num_batches += 1

        # Dynamic logging interval based on global batch size.
        batch_size_per_gpu = input_data.size(0)
        global_batch_size = batch_size_per_gpu * dist.get_world_size()
        log_interval = max(1, 1600 // max(global_batch_size, 1))

        if is_main_process(rank) and idx % log_interval == 0:
            samples_seen = (epoch * len(dataloader) + idx) * batch_size_per_gpu * dist.get_world_size()
            _gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            try:
                iter_log = {
                    "train/loss_bev": loss_total_bev.item(),
                    "train/loss_offset": loss_offset.item(),
                    "train/loss_2d": loss_total_2d.item(),
                    "train/loss_height": loss_height.item(),
                    "train/loss_total": loss_total.item(),
                    "train/grad_norm": _gn,
                    "train/iter": epoch * len(dataloader) + idx,
                    "train/samples_seen": samples_seen,
                    "train/step_lr": optimizer.param_groups[0]['lr'],
                }
                for key, value in loss_details.items():
                    if torch.is_tensor(value):
                        iter_log[f"train/loss_detail/{key}"] = value.detach().item()
                    else:
                        iter_log[f"train/loss_detail/{key}"] = float(value)
                wandb.log(iter_log)
            except Exception as e:
                print(f"[Rank {rank}] wandb iter logging failed: {e}")
            pbar.set_postfix({
                'bev': f'{loss_total_bev.item():.4f}',
                'total': f'{loss_total.item():.4f}'
            })

        global_step += 1

        if idx % 300 == 0:
            dist.barrier()

    for t in [total_loss_bev, total_loss_offset, total_loss_2d, total_loss_height]:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    for t in total_loss_detail.values():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size()
    total_batches = num_batches * world_size

    avg_losses = {
        "train/epoch_loss_bev": (total_loss_bev / total_batches).item(),
        "train/epoch_loss_offset": (total_loss_offset / total_batches).item(),
        "train/epoch_loss_2d": (total_loss_2d / total_batches).item(),
        "train/epoch_loss_height": (total_loss_height / total_batches).item(),
        "train/epoch_loss_total": ((total_loss_bev + total_loss_offset + total_loss_2d + total_loss_height) / total_batches).item(),
        "train/epoch_grad_norm": total_grad_norm / max(num_batches, 1),
        "train/epoch": epoch
    }
    for key, value in total_loss_detail.items():
        avg_losses[f"train/epoch_loss_detail/{key}"] = (value / total_batches).item()

    if is_main_process(rank):
        try:
            wandb.log(avg_losses)
        except Exception as e:
            print(f"[Rank {rank}] wandb epoch logging failed: {e}")

    return avg_losses, global_step


@torch.no_grad()
def validate(rank, world_size, model, val_loader, val_dataset, configs, epoch):
    """Apollo validation — distributed across all GPUs, results gathered at rank 0.

    val_dataset must be an Apollo_dataset_with_offset_val instance; its
    _name_to_idx dict maps (folder, filename) -> info_dict index.

    Coordinate note:
        bev_instance2points_with_offset_z output:
          lane[0] = longitudinal (y_lon)
          lane[1] = lateral with internal y*=-1 applied  (== -x_lat)
        Apollo GT persformer format col0 = -x_lat, so lane[1] is used as-is
        (no sign flip needed, unlike OpenLane which uses -1*lane[1]).
    """
    model.eval()

    x_range = configs.x_range
    meter_per_pixel = configs.meter_per_pixel

    post_conf = getattr(configs, 'post_conf', -1.7)
    post_emb_margin = getattr(configs, 'post_emb_margin', 6.0)
    post_min_cluster_size = getattr(configs, 'post_min_cluster_size', 15)

    height_abs_errors = []
    bench_data = []

    for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}", disable=not is_main_process(rank)):
        image, bn_name, intrinsic, road2cam, heightmap_gt = batch
        image = image.to(rank, non_blocking=True)
        intrinsic = intrinsic.to(rank, non_blocking=True)
        road2cam = road2cam.to(rank, non_blocking=True)

        model_out = model.module.model(image, intrinsic, road2cam)

        pred_ = model_out[0]
        height_ = model_out[1]

        seg = pred_[0].cpu().numpy()
        embedding = pred_[1].cpu().numpy()
        offset_y = torch.sigmoid(pred_[2]).cpu().numpy()

        if isinstance(height_, list):
            height_ = height_[-1]
        if height_.dim() == 3:
            height_ = height_.unsqueeze(1)
        height = height_.cpu().numpy()

        # heightmap_gt shape: (B, 2, H, W) — ch0: height, ch1: binary mask
        gt_height_np = heightmap_gt[:, 0].numpy()
        gt_mask_np = heightmap_gt[:, 1].numpy()

        batch_size = seg.shape[0]
        folders = bn_name[0]
        filenames = bn_name[1]

        for idx in range(batch_size):
            valid = gt_mask_np[idx] > 0.5
            if valid.sum() > 0:
                pred_h = height[idx, 0]
                gt_h = gt_height_np[idx]
                abs_err = np.abs(pred_h[valid] - gt_h[valid])
                height_abs_errors.append(abs_err)

            ms = seg[idx:idx + 1]
            me = embedding[idx:idx + 1]
            moffset = offset_y[idx:idx + 1]
            z = height[idx:idx + 1]

            prediction = (ms, me)
            canvas, _ = embedding_post(
                prediction, conf=post_conf,
                emb_margin=post_emb_margin,
                min_cluster_size=post_min_cluster_size,
                canvas_color=False)

            lines = bev_instance2points_with_offset_z(
                canvas, max_x=x_range[1],
                meter_per_pixal=(meter_per_pixel, meter_per_pixel),
                offset_y=moffset[0][0], Z=z[0][0])

            folder = folders[idx]
            filename = filenames[idx]
            list_idx = val_dataset._name_to_idx.get((folder, filename), None)
            if list_idx is None:
                print(f"[val] GT not found: {folder}/{filename}")
                continue
            info_dict = val_dataset.cnt_list[list_idx]
            frame_lanes_gt = get_gt_lanes_apollo(info_dict)

            frame_lanes_pred = []
            for lane in lines:
                pred_in_persformer = np.array([lane[1], lane[0], lane[2]])
                frame_lanes_pred.append(pred_in_persformer.T.tolist())

            bench_data.append((frame_lanes_pred, frame_lanes_gt))

    all_bench_data = [None] * world_size
    dist.all_gather_object(all_bench_data, bench_data)

    all_height_errors = [None] * world_size
    dist.all_gather_object(all_height_errors, height_abs_errors)

    if not is_main_process(rank):
        return {}

    lane_eval = LaneEval()
    for rank_bench in all_bench_data:
        for frame_lanes_pred, frame_lanes_gt in rank_bench:
            lane_eval.bench_all(frame_lanes_pred, frame_lanes_gt)

    print(f"\n[Epoch {epoch}] Validation Results:")
    results = lane_eval.show()

    combined_errors = [err for rank_errors in all_height_errors for err in rank_errors]
    if combined_errors:
        all_abs_err = np.concatenate(combined_errors)
        results['height_mae']     = float(np.mean(all_abs_err))
        results['height_rmse']    = float(np.sqrt(np.mean(all_abs_err ** 2)))
        results['height_acc_005'] = float(np.mean(all_abs_err < 0.05))
        results['height_acc_010'] = float(np.mean(all_abs_err < 0.10))
        results['height_acc_020'] = float(np.mean(all_abs_err < 0.20))
    else:
        results['height_mae']     = 0.0
        results['height_rmse']    = 0.0
        results['height_acc_005'] = 0.0
        results['height_acc_010'] = 0.0
        results['height_acc_020'] = 0.0

    print(f"  Height MAE:  {results['height_mae']:.4f}")
    print(f"  Height RMSE: {results['height_rmse']:.4f}")
    print(f"  Acc@0.05:    {results['height_acc_005']:.4f}")
    print(f"  Acc@0.10:    {results['height_acc_010']:.4f}")
    print(f"  Acc@0.20:    {results['height_acc_020']:.4f}")

    try:
        wandb.log({
            "val/F1": results.get('f1_score', 0),
            "val/Precision": results.get('precision', 0),
            "val/Recall": results.get('recall', 0),
            "val/x_error_close": results.get('x_error_close', 0),
            "val/x_error_far": results.get('x_error_far', 0),
            "val/z_error_close": results.get('z_error_close', 0),
            "val/z_error_far": results.get('z_error_far', 0),
            "val/height_mae": results.get('height_mae', 0),
            "val/height_rmse": results.get('height_rmse', 0),
            "val/height_acc_005": results.get('height_acc_005', 0),
            "val/height_acc_010": results.get('height_acc_010', 0),
            "val/height_acc_020": results.get('height_acc_020', 0),
            "val/epoch": epoch
        })
    except Exception as e:
        print(f"[Rank {rank}] wandb validation logging failed: {e}")

    dist.barrier()
    return results


def setup(rank, world_size, master_port='12356'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=10))
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def worker_function(rank, world_size, config_file, checkpoint_path=None, use_amp=False,
                    wandb_project="hsdflane_apollo", wandb_name=None,
                    master_port='12356', skip_val=False, warmup_steps=1000):
    setup(rank, world_size, master_port)

    if is_main_process(rank):
        print(f'Using {world_size} GPUs: ' + ','.join([str(i) for i in range(world_size)]))

    configs = load_config_module(config_file)

    resume_run_id = None
    if checkpoint_path and is_main_process(rank):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            resume_run_id = ckpt.get('wandb_run_id', None)
            if resume_run_id:
                print(f"Found WandB run ID in checkpoint: {resume_run_id}")
        except Exception as e:
            print(f"Could not load run_id from checkpoint: {e}")

    if is_main_process(rank):
        try:
            wandb_key = os.environ.get("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key)
            else:
                wandb.login()
            split_tag = getattr(configs, 'SPLIT_TYPE', 'standard')
            run_name = wandb_name or f"hsdflane_apollo_{split_tag}_{configs.epochs}ep_ddp{world_size}gpu"

            if resume_run_id:
                wandb.init(project=wandb_project, name=run_name,
                           group='distributed_training',
                           id=resume_run_id, resume="allow")
                print(f"✓ wandb resumed: run_id={resume_run_id}")
            else:
                wandb.init(project=wandb_project, name=run_name,
                           group='distributed_training')
                print(f"✓ wandb initialized: project={wandb_project}, name={run_name}")
        except Exception as e:
            print(f"✗ wandb init failed: {e}")

    model = configs.model()
    CML = getattr(configs, 'Combine_Model_and_Loss', None)
    if CML is None:
        raise RuntimeError("Combine_Model_and_Loss not defined in config.")
    model = CML(model)
    if hasattr(model, 'return_loss_details'):
        model.return_loss_details = True
    model = model.to(rank)
    find_unused = getattr(configs, 'find_unused_parameters', False)
    model = DDP(model, device_ids=[rank], find_unused_parameters=find_unused)

    optimizer_factory = getattr(configs, 'optimizer_factory', None)
    if optimizer_factory is not None:
        optimizer = optimizer_factory(
            model,
            base_lr=configs.optimizer_params.get('lr', 2.5e-4),
            head_lr=configs.optimizer_params.get('head_lr',
                      configs.optimizer_params.get('lr', 2.5e-4)),
            weight_decay=configs.optimizer_params.get('weight_decay', 1e-2),
            optimizer_cls=configs.optimizer,
        )
    else:
        optimizer = configs.optimizer(
            filter(lambda p: p.requires_grad, model.parameters()),
            **configs.optimizer_params
        )

    base_lrs = [group['lr'] for group in optimizer.param_groups]
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)
    scaler = GradScaler() if use_amp else None

    start_epoch = 0
    wandb_run_id = None
    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            start_epoch, wandb_run_id = resume_training(
                checkpoint_path, model.module, optimizer, scheduler)
            if is_main_process(rank):
                print(f"Resumed from epoch {start_epoch - 1}, "
                      f"starting training at epoch {start_epoch}")
        else:
            load_checkpoint(checkpoint_path, model.module, None)

    if is_main_process(rank) and wandb_run_id is None:
        try:
            wandb_run_id = wandb.run.id
            print(f"New WandB run ID: {wandb_run_id}")
        except Exception:
            pass

    Dataset = getattr(configs, "train_dataset", None)
    if Dataset is None:
        Dataset = configs.training_dataset

    train_dataset = Dataset()
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    loader_args = dict(configs.loader_args)
    loader_args.pop('shuffle', None)

    train_loader = DataLoader(
        train_dataset,
        **loader_args,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=4,
    )

    global_step = start_epoch * len(train_loader)

    # Apollo val needs val_dataset instance for _name_to_idx lookup on all ranks
    val_dataset_inst = None
    val_loader = None
    if not skip_val:
        val_dataset_inst = configs.val_dataset()
        _val_args = getattr(configs, 'val_loader_args', {})
        val_sampler = DistributedSampler(val_dataset_inst, num_replicas=world_size, rank=rank,
                                         shuffle=False, drop_last=False)
        val_loader = DataLoader(
            val_dataset_inst,
            batch_size=_val_args.get('batch_size', 4),
            num_workers=_val_args.get('num_workers', 4),
            shuffle=False,
            pin_memory=True,
            sampler=val_sampler,
            persistent_workers=False,
        )

    torch.cuda.empty_cache()

    best_f1 = 0.0

    for epoch in range(start_epoch, configs.epochs):
        dist.barrier()
        train_sampler.set_epoch(epoch)

        train_losses, global_step = train_epoch(
            rank, model, train_loader, optimizer, scaler, epoch,
            use_amp, global_step, warmup_steps, base_lrs,
        )
        scheduler.step()

        if is_main_process(rank):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: LR = {current_lr:.6e}")
            try:
                wandb.log({"train/learning_rate": current_lr, "train/epoch": epoch})
            except Exception:
                pass

        # Save checkpoint (rank 0 only)
        if is_main_process(rank):
            save_model_dp(model, optimizer, configs.model_save_path, f'ep{epoch:03d}.pth',
                          epoch=epoch, scheduler=scheduler, wandb_run_id=wandb_run_id)
            save_model_dp(model, optimizer, configs.model_save_path, 'latest.pth',
                          epoch=epoch, scheduler=scheduler, wandb_run_id=wandb_run_id)

        if skip_val:
            if is_main_process(rank):
                print(f"Epoch {epoch} completed (val skipped). "
                      f"Train Loss: {train_losses['train/epoch_loss_bev']:.4f}")
        else:
            val_results = validate(rank, world_size, model, val_loader, val_dataset_inst, configs, epoch)

            if is_main_process(rank):
                current_f1 = val_results.get('f1_score', 0)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    save_model_dp(model, optimizer, configs.model_save_path, 'best.pth',
                                  epoch=epoch, scheduler=scheduler,
                                  wandb_run_id=wandb_run_id)
                    print(f"New best model saved with F1: {best_f1:.4f}")
                print(f"Epoch {epoch} completed. "
                      f"Train Loss: {train_losses['train/epoch_loss_bev']:.4f}, "
                      f"Val F1: {current_f1:.4f}")

    if is_main_process(rank):
        try:
            wandb.finish()
            print("✓ wandb session closed")
        except Exception as e:
            print(f"✗ wandb finish failed: {e}")

    cleanup()


def main():
    parser = argparse.ArgumentParser(description='HSDF-Lane Apollo DDP Training')
    parser.add_argument('--config', type=str,
                        default='./tools/hsdflane_apollo_config.py',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for resume training')
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable Automatic Mixed Precision training')
    parser.add_argument('--port', type=str, default='12356',
                        help='Master port for distributed training '
                             '(default 12356: avoids collision with OpenLane training)')
    parser.add_argument('--wandb_project', type=str, default='hsdflane_apollo',
                        help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='wandb run name')
    parser.add_argument('--skip_val', action='store_true',
                        help='Skip validation during training')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of global warmup iterations (linear warmup)')
    args = parser.parse_args()

    world_size = args.gpus if args.gpus else torch.cuda.device_count()
    assert world_size > 0, "No GPU available!"

    print(f"Starting HSDF-Lane Apollo DDP training with {world_size} GPUs")
    print(f"Config: {args.config}")
    print(f"AMP: {args.amp}")
    print(f"Warmup steps: {args.warmup_steps}")

    mp.spawn(
        worker_function,
        args=(world_size, args.config, args.checkpoint, args.amp,
              args.wandb_project, args.wandb_name, args.port,
              args.skip_val, args.warmup_steps),
        nprocs=world_size,
        join=True,
    )


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
