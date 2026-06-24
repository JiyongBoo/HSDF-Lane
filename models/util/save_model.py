import os
import pathlib
import torch
import torch.distributed as dist
import time


def save_model(net, optimizer, save_path, name):
    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        net = net.module
    if dist.get_rank() == 0:
        model_state_dict = net.state_dict()
        state = {'models': model_state_dict, 'optimizer': optimizer.state_dict() if optimizer else None}
        # state = {'models': model_state_dict}
        if not os.path.exists(save_path):
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            # os.mkdir(save_path)
        model_path = os.path.join(save_path, name)
        torch.save(state, model_path)


def save_model_dp(net, optimizer, save_path, name, epoch=None, scheduler=None, wandb_run_id=None):
    """ save current models
    """
    save_path = os.path.join('./train_results', os.path.basename(save_path))
    if not os.path.exists(save_path):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(save_path, name)
    model_to_save = net.module if hasattr(net, 'module') else net
    torch.save({
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "wandb_run_id": wandb_run_id,
    }, model_path)
    print("Model saved as %s" % model_path)
