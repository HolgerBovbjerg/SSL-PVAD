import os
import random
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch import nn, optim

import wandb


def freeze_model_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        count += sum(f.endswith(extension) for f in files)
    return count


def nearest_interp(x_interpolate, x_original, y_original):
    x_distance = torch.abs(x_original - x_interpolate[:, None])
    y_interpolate = y_original[torch.argmin(x_distance, dim=1)]
    return y_interpolate


def compute_model_size(model):
    """
    Computes size of pytorch model in MB
    :param model: PyTorch model
    :return: size of model in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2

    return size_all_mb


def count_parameters(model, trainable: bool = False) -> int:
    """
    Computes size of pytorch model in MB
    :param model: PyTorch model
    :param trainable: If true, only trainable parameters are counted
    :return: number of parameters in model
    """
    if trainable:
        count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    else:
        count = sum(p.numel() for p in model.parameters())
    return count


def seed_everything(seed: int) -> None:
    """
    Set manual seed of python, numpy, pytorch and cuda
    :param seed: Supplied seed.
    :return: None
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'Set {seed=}')


def log(log_dict: dict, step: int, config: dict) -> None:
    """Handles logging for metric tracking server, local disk and stdout.
    Args:
        log_dict (dict): Log metric dict.
        step (int): Current step.
        config (dict): Config dict.
    """

    # send logs to wandb tracking server
    if config["exp"]["wandb"]:
        wandb.log(log_dict, step=step)

    log_message = f"Step: {step} | " + " | ".join(log_dict)

    # write logs to disk
    if config["exp"]["log_to_file"]:
        log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

        with open(log_file, "a+", encoding="utf8") as file:
            file.write(log_message + "\n")

    # show logs in stdout
    if config["exp"]["log_to_stdout"]:
        log_dict_formatted = []
        for key, value in log_dict.items():
            if key in ["loss", "lr"]:
                log_dict_formatted.append(f"{key}: {value:.3e}")
            elif isinstance(value, float):
                log_dict_formatted.append(f"{key}: {value:.3f}")
            else:
                log_dict_formatted.append(f"{key}: {value}")
        print(f"Step: {step} | " + " | ".join(log_dict_formatted))


def save_model(epoch: int, score: float, save_path: str, net: nn.Module, optimizer: Optional[optim.Optimizer] = None,
               log_file: Optional[str] = None, **kwargs) -> None:
    """Saves checkpoint.
    Args:
        epoch (int): Current epoch.
        score (float): Validation accuracy.
        save_path (str): Checkpoint path.
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer, optional): Optimizer. Defaults to None.
        log_file (str, optional): Log file. Defaults to None.
    """

    ckpt_dict = {
        "epoch": epoch,
        "score": score,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else optimizer
    }
    ckpt_dict.update(kwargs)

    torch.save(ckpt_dict, save_path)

    log_message = f"Saved {save_path} with score {score}."
    print(log_message)

    if log_file is not None:
        with open(log_file, "a+", encoding="utf8") as file:
            file.write(log_message + "\n")


def calc_step(epoch: int, n_batches: int, batch_index: int) -> int:
    """Calculates current step.
    Args:
        epoch (int): Current epoch.
        n_batches (int): Number of batches in dataloader.
        batch_index (int): Current batch index.
    Returns:
        int: Current step.
    """
    return (epoch - 1) * n_batches + (1 + batch_index)


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c", label="max gradient")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b", label="mean gradient")
    ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k", label="zero gradient")
    ax.set_xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    ax.set_xlim(left=0, right=len(ave_grads))
    #ax.set_ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("Gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig, ax
