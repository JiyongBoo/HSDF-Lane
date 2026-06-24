from typing import Dict
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler


def load_checkpoint(checkpoint, net, optimizer=None, map_loc="cuda"):
    sd = torch.load(checkpoint, map_location=map_loc)
    # DDP 체크포인트의 'module.' 접두사 제거 (single-GPU 호환)
    state_dict = sd['model_state']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    if optimizer and sd['optimizer_state']:
        optimizer.load_state_dict(sd['optimizer_state'])
    return sd


def resume_training(checkpoint: Dict,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: _LRScheduler):
    # Load checkpoint
    sd = load_checkpoint(checkpoint, model, optimizer)
    
    # Load scheduler state if exists
    if 'scheduler' in sd and sd['scheduler'] is not None:
        scheduler.load_state_dict(sd['scheduler'])
    
    # Return starting epoch and wandb_run_id
    start_epoch = sd.get('epoch', 0) + 1  # Next epoch to train
    wandb_run_id = sd.get('wandb_run_id', None)
    return start_epoch, wandb_run_id


def load_model(model, model_state_file):
    pretrained_dict = torch.load(model_state_file, map_location='cpu')
    if 'model_state' in pretrained_dict:
        pretrained_dict1 = pretrained_dict['model_state']
    elif 'state_dict' in pretrained_dict:
        pretrained_dict1 = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict1.items()
                       if k[6:] in model_dict.keys()}
    if len(pretrained_dict) == 0:
        pretrained_dict = {k: v for k, v in pretrained_dict1.items()
                           if k in model_dict.keys()}
    count = 0
    for k in model_dict.keys():
        if k not in pretrained_dict.keys():
            count += 1
            print(k, "is not in pretrained_dict")
    print(count)
    model_dict.update(pretrained_dict)
    # meshgrid/expand 등으로 생성된 버퍼는 메모리를 공유(view)하여
    # load_state_dict에서 "more than one element refers to a single memory location" 에러 발생.
    # clone()으로 독립 메모리를 보장한다.
    model_dict = {k: v.clone() if isinstance(v, torch.Tensor) else v
                  for k, v in model_dict.items()}
    model.load_state_dict(model_dict)
    return model
