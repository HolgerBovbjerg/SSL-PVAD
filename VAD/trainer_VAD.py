import os
import time
from typing import Callable
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, roc_auc_score

from common.misc import log, save_model, unfreeze_model_parameters, count_parameters
from common.metrics import wandb_log_confusion_matrix


def train_single_batch(model: nn.Module, features: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor,
                       optimizer: optim.Optimizer, criterion: Callable, device: torch.device):
    features, targets, lengths = features.to(device), targets.to(device), lengths.to(device)

    optimizer.zero_grad()

    predictions, lengths = model(features, lengths)

    loss = criterion(predictions.transpose(-1, -2), targets)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    outputs_list = []
    targets_list = []

    for j in range(predictions.size(0)):
        probabilities = torch.nn.functional.softmax(predictions[j][:lengths[j]], dim=-1)
        outputs_list.append(probabilities.detach().cpu().numpy())
        targets_list.append(targets[j][:lengths[j]].cpu().numpy())

    targets_all = np.concatenate(targets_list)
    outputs_all = np.concatenate(outputs_list)
    targets_one_hot = np.eye(2)[targets_all]

    cm = confusion_matrix(targets_all, outputs_all.argmax(-1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    accuracy = cm.diagonal()
    average_precision = average_precision_score(targets_one_hot, outputs_all, average=None)
    aucroc = roc_auc_score(targets_one_hot, outputs_all, average=None)
    mean_average_precision = average_precision_score(targets_one_hot, outputs_all, average="micro")

    metric_scores = {"train_accuracy_ns": accuracy[0],
                     "train_accuracy_s": accuracy[1],
                     "train_averageprecision_ns": average_precision[0],
                     "train_averageprecision_s": average_precision[1],
                     "train_auroc_ns": aucroc[0],
                     "train_auroc_s": aucroc[1],
                     "train_mAP": mean_average_precision}
    return loss.item(), metric_scores


def train_single_epoch(model: nn.Module, epoch: int, step: int, augmentor: Callable, scheduler: dict,
                       train_loader: DataLoader, optimizer: optim.Optimizer, criterion: Callable,
                       device: torch.device, config):
    running_loss = 0.
    batch_step = 0
    epoch_start_time = time.time()

    model.train()
    batch_start_time = time.time()
    for batch_index, data in enumerate(train_loader):
        data_load_time = time.time() - batch_start_time
        features, targets, _, lengths, _ = data

        if augmentor:
            for i, feature in enumerate(features):
                features[i, :lengths[i]] = augmentor(feature[:lengths[i]])

        loss, metric_scores = train_single_batch(model, features, targets, lengths, optimizer,
                                                 criterion, device)
        running_loss += loss

        if not step % config["exp"]["log_freq"]:
            log_dict = {"epoch": epoch,
                        "train_loss": loss,
                        "lr": optimizer.param_groups[0]["lr"]}
            log(log_dict, step, config)

        if scheduler:
            scheduler.step()

        log_dict = {"epoch": epoch,
                    "train_loss": loss,
                    "batch": batch_index,
                    "time_per_batch": time.time() - batch_start_time,
                    "data_load_time": data_load_time}
        log_dict.update(metric_scores)
        log(log_dict, step, config)

        batch_step += 1
        step += 1

        batch_start_time = time.time()

    log_dict = {"epoch": epoch,
                "time_per_epoch": time.time() - epoch_start_time,
                "avg_train_loss": running_loss / batch_step}

    return step, log_dict


def train(model: nn.Module, optimizer: optim.Optimizer, criterion: Callable, scheduler: dict, train_loader: DataLoader,
          validation_loader: DataLoader, config: dict, augmentor: Callable = None):
    epochs = config["hparams"]["n_epochs"]
    step = 0
    best_score = 0.
    device = config["exp"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

    frozen = False
    if "checkpoint" in config["hparams"]["model"]:
        if config["hparams"]["model"]["checkpoint"]["freeze_first_k_epochs"]:
            frozen = True
            freeze_first_k_epochs = config["hparams"]["model"]["checkpoint"]["freeze_first_k_epochs"]

    #######################
    # Training start
    #######################
    for epoch in tqdm(range(epochs), unit="Epoch", position=0, leave=True):
        if frozen and epoch < freeze_first_k_epochs:
            unfreeze_model_parameters(model.encoder)
            print(f"{count_parameters(model.encoder)} parameters unfrozen. "
                  f"{count_parameters(model, trainable=True)} trainable parameters.")
            frozen = False
            
        step, log_dict = train_single_epoch(model=model, epoch=epoch, step=step, optimizer=optimizer,
                                            augmentor=augmentor, scheduler=scheduler, train_loader=train_loader,
                                            criterion=criterion, device=device, config=config)
        log(log_dict, step, config)

        if not epoch % config["exp"]["val_freq"]:
            avg_val_loss, metric_scores, conf_matrix = evaluate(model, criterion, validation_loader, device)
            log_dict = {"epoch": epoch, "avg_val_loss": avg_val_loss}

            metric_scores = {"avg_val_" + k: val for k, val in metric_scores.items()}

            log_dict.update(metric_scores)
            log(log_dict, step, config)

            if config["exp"]["wandb"]:
                wandb.log({"val_confusion_matrix": wandb_log_confusion_matrix(conf_matrix,
                                                                              class_names=["ns", "s"])},
                          step=step)

            # save best val ckpt
            score = metric_scores["avg_val_mAP"]
            if score >= best_score:
                best_score = score
                save_path = os.path.join(config["exp"]["save_dir"], "best.pt")
                save_model(epoch, score, save_path, model, optimizer, log_file)

    ###########################
    # Training complete
    ###########################

    avg_val_loss, metric_scores, conf_matrix = evaluate(model, criterion, validation_loader, device)
    log_dict = {"epoch": epoch, "avg_val_loss": avg_val_loss}
    metric_scores = {"avg_val_" + k: val for k, val in metric_scores.items()}
    log_dict.update(metric_scores)
    log(log_dict, step, config)

    if config["exp"]["wandb"]:
        wandb.log({"val_confusion_matrix": wandb_log_confusion_matrix(conf_matrix,
                                                                      class_names=["ns", "s"])},
                  step=step)

    # save final ckpt
    score = metric_scores["avg_val_mAP"]
    save_path = os.path.join(config["exp"]["save_dir"], "last.pt")
    save_model(epoch, score, save_path, model, optimizer, log_file)

    return step


@torch.no_grad()
def evaluate(model: nn.Module, criterion: Callable, data_loader: DataLoader, device: torch.device):
    model.eval()

    running_loss = 0.
    batch_step = 0

    outputs_list = []
    targets_list = []

    for batch_index, data in enumerate(data_loader):
        features, targets, _, lengths, _ = data
        features, targets, lengths = features.to(device), targets.to(device), lengths.to(device)
        targets[targets == 2] = 1.

        predictions, lengths = model(features, lengths)
        loss = criterion(predictions.transpose(-1, -2), targets)

        for j in range(predictions.size(0)):
            probabilities = torch.nn.functional.softmax(predictions[j][:lengths[j]], dim=-1)
            outputs_list.append(probabilities.cpu().numpy())
            targets_list.append(targets[j][:lengths[j]].cpu().numpy())

        running_loss += loss.item()
        batch_step += 1

    model.train()

    avg_loss = running_loss / batch_step

    targets_all = np.concatenate(targets_list)
    outputs_all = np.concatenate(outputs_list)
    targets_one_hot = np.eye(2)[targets_all]

    cm = confusion_matrix(targets_all, outputs_all.argmax(-1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    accuracy = cm.diagonal()
    average_precision = average_precision_score(targets_one_hot, outputs_all, average=None)
    aucroc = roc_auc_score(targets_one_hot, outputs_all, average=None)
    mean_average_precision = average_precision_score(targets_one_hot, outputs_all, average="micro")

    metric_scores = {"accuracy_ns": accuracy[0],
                     "accuracy_s": accuracy[1],
                     "averageprecision_ns": average_precision[0],
                     "averageprecision_s": average_precision[1],
                     "auroc_ns": aucroc[0],
                     "auroc_s": aucroc[1],
                     "mAP": mean_average_precision}

    return avg_loss, metric_scores, cm
