import os
import time
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from common.metrics import wandb_log_confusion_matrix
from common.misc import log, save_model


def non_padding_instance_mean_std(input_tensor, lengths):
    tensors = [tensor[:length] for tensor, length in zip(input_tensor, lengths)]
    means = [tensor.mean() for tensor in tensors]
    stds = [tensor.std() for tensor in tensors]
    return torch.tensor(stds), torch.tensor(means)


def train_single_batch(model: nn.Module, features: torch.Tensor, targets: torch.Tensor,
                       similarities: torch.Tensor, lengths: torch.Tensor,
                       optimizer: optim.Optimizer, criterion: Callable, device: torch.device, batch_index,
                       epoch, config, scheduler):
    features, targets, similarities, lengths = features.to(device), targets.to(device), \
        similarities.to(device), lengths.to(device)

    predictions, lengths = model(features, similarities, lengths)

    loss = criterion(predictions.transpose(-1, -2), targets) / config["hparams"]["loss"]["accumulation_steps"]
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if ((batch_index + 1) % config["hparams"]["loss"]["accumulation_steps"] == 0):
        optimizer.step()
        optimizer.zero_grad()

    outputs_list = []
    targets_list = []

    for j in range(predictions.size(0)):
        probabilities = torch.nn.functional.softmax(predictions[j][:lengths[j]], dim=-1)
        outputs_list.append(probabilities.detach().cpu().numpy())
        targets_list.append(targets[j][:lengths[j]].cpu().numpy())

    targets_all = np.concatenate(targets_list)
    outputs_all = np.concatenate(outputs_list)
    targets_one_hot = np.eye(3)[targets_all]

    cm = confusion_matrix(targets_all, outputs_all.argmax(-1), labels=[0, 1, 2])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    accuracy = cm.diagonal()
    average_precision = average_precision_score(targets_one_hot, outputs_all, average=None)
    # aucroc = roc_auc_score(targets_one_hot, outputs_all, average=None)
    mean_average_precision = average_precision_score(targets_one_hot, outputs_all, average="micro")

    metric_scores = {"train_accuracy_ns": accuracy[0],
                     "train_accuracy_tss": accuracy[1],
                     "train_accuracy_ntss": accuracy[2],
                     "train_averageprecision_ns": average_precision[0],
                     "train_averageprecision_tss": average_precision[1],
                     "train_averageprecision_ntss": average_precision[2],
                     # "train_auroc_ns": aucroc[0],
                     # "train_auroc_tss": aucroc[1],
                     # "train_auroc_ntss": aucroc[2],
                     "train_mAP": mean_average_precision}
    metric_scores = {k: metric_scores[k] if not np.isnan(val) else 0. for k, val in metric_scores.items()}
    return loss.item(), metric_scores


def train_single_epoch(model: nn.Module, epoch: int, step: int, augmentor: Callable, scheduler: dict,
                       train_loader: DataLoader, optimizer: optim.Optimizer, criterion: Callable,
                       device: torch.device, config):
    label2number = config["data"]["labels"]
    p_target_free = config["data"]["p_target_free"]
    p_enrollment_free = config["data"]["p_enrollment_free"]
    running_loss = 0.
    batch_accum_loss = 0.
    batch_step = 0
    metric_scores_accum = None
    epoch_start_time = time.time()

    model.train()
    batch_start_time = time.time()
    for batch_index, data in enumerate(train_loader):
        data_load_time = time.time() - batch_start_time
        features, targets, similarities, lengths, target_speaker_ids = data
        batch_size = features.size(0)

        if augmentor:
            for i, feature in enumerate(features):
                features[i, :lengths[i]] = augmentor(feature[:lengths[i]])

        # target_free = torch.rand(batch_size) < p_target_free
        # target_free_mask = torch.logical_and(targets == label2number["tss"], target_free.unsqueeze(1))
        # targets = targets.masked_fill(target_free_mask, label2number["ntss"])
        # for index, val in enumerate(target_free):
        #     if val:
        #         target_speaker_id = target_speaker_ids[index]
        #         other_target_speaker_ids = [i for i in target_speaker_ids if i != target_speaker_id]
        #         new_target_speaker_id = other_target_speaker_ids[random.randint(0, len(other_target_speaker_ids)-1)]
        #         target_speaker_ids[index] = new_target_speaker_id
        #         speaker_embeddings[index] = speaker_embeddings[target_speaker_ids.index(new_target_speaker_id)]
        #
        # enrollment_free = torch.rand(batch_size) < p_enrollment_free
        # speaker_embeddings[enrollment_free] = 0.
        # enrollment_free_mask = torch.logical_and(targets == label2number["ntss"], enrollment_free.unsqueeze(1))
        # targets = targets.masked_fill(enrollment_free_mask, label2number["tss"])

        loss, metric_scores = train_single_batch(model=model, features=features, targets=targets,
                                                 similarities=similarities, lengths=lengths,
                                                 optimizer=optimizer, criterion=criterion, device=device,
                                                 batch_index=batch_index, epoch=epoch, config=config,
                                                 scheduler=scheduler)

        batch_accum_loss += loss
        if metric_scores_accum:
            metric_scores_accum = {k: metric_scores_accum[k] + val / config["hparams"]["loss"]["accumulation_steps"] for
                                   k, val in metric_scores.items()}
        else:
            metric_scores_accum = {k: val / config["hparams"]["loss"]["accumulation_steps"] for k, val in
                                   metric_scores.items()}

        if ((batch_index + 1) % config["hparams"]["loss"]["accumulation_steps"] == 0):
            running_loss += batch_accum_loss

            if not step % config["exp"]["log_freq"]:
                log_dict = {"epoch": epoch, "train_loss": batch_accum_loss, "lr": optimizer.param_groups[0]["lr"]}
                log(log_dict, step, config)

                log_dict.update({"batch": batch_index,
                                 "time_per_batch": time.time() - batch_start_time,
                                 "data_load_time": data_load_time})
                log_dict.update(metric_scores_accum)
                log(log_dict, step, config)

            if scheduler:
                scheduler.step()

            batch_accum_loss = 0.
            metric_scores_accum = {k: 0 for k in metric_scores.keys()}
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
    label2number = config["data"]["labels"]

    #######################
    # Training start
    #######################
    for epoch in tqdm(range(epochs), unit="Epoch", position=0, leave=True):
        step, log_dict = train_single_epoch(model=model, epoch=epoch, step=step, optimizer=optimizer,
                                            augmentor=augmentor, scheduler=scheduler, train_loader=train_loader,
                                            criterion=criterion, device=device, config=config)
        log(log_dict, step, config)

        if not epoch % config["exp"]["val_freq"]:
            avg_val_loss, metric_scores, conf_matrix = evaluate(model=model, criterion=criterion,
                                                                data_loader=validation_loader, device=device)
            log_dict = {"epoch": epoch, "avg_val_loss": avg_val_loss}
            metric_scores = {"avg_val_" + k: val for k, val in metric_scores.items()}
            log_dict.update(metric_scores)
            log(log_dict, step, config)

            if config["exp"]["wandb"]:
                wandb.log({"val_confusion_matrix": wandb_log_confusion_matrix(conf_matrix,
                                                                              class_names=list(label2number.keys()))},
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

    avg_val_loss, metric_scores, conf_matrix = evaluate(model=model, criterion=criterion, data_loader=validation_loader,
                                                        device=device)
    log_dict = {"epoch": epoch, "avg_val_loss": avg_val_loss}
    metric_scores = {"avg_val_" + k: val for k, val in metric_scores.items()}
    log_dict.update(metric_scores)
    log_dict.update(metric_scores)
    log(log_dict, step, config)

    if config["exp"]["wandb"]:
        wandb.log({"val_confusion_matrix": wandb_log_confusion_matrix(conf_matrix,
                                                                      class_names=list(label2number.keys()))},
                  step=step)

    # save final ckpt
    score = metric_scores["avg_val_mAP"]
    save_path = os.path.join(config["exp"]["save_dir"], "last.pt")
    save_model(epoch, score, save_path, model, optimizer, log_file)

    return step


@torch.no_grad()
def evaluate(model: nn.Module, criterion: Callable, data_loader: DataLoader,
             device: torch.device):
    model.eval()

    running_loss = 0.
    batch_step = 0
    outputs_list = []
    targets_list = []
    for batch_index, data in enumerate(data_loader):
        features, targets, similarities, lengths, _ = data
        features, targets, similarities, lengths = features.to(device), targets.to(device), \
            similarities.to(device), lengths.to(device)

        predictions, lengths = model(features, similarities, lengths)
        loss = criterion(predictions.transpose(1, 2), targets)

        for j in range(predictions.size(0)):
            probabilities = torch.nn.functional.softmax(predictions[j][:lengths[j]], dim=-1)
            outputs_list.append(probabilities.detach().cpu().numpy())
            targets_list.append(targets[j][:lengths[j]].cpu().numpy())

        running_loss += loss.item()
        batch_step += 1

    model.train()

    avg_loss = running_loss / batch_step

    targets_all = np.concatenate(targets_list)
    outputs_all = np.concatenate(outputs_list)
    targets_one_hot = np.eye(3)[targets_all]

    cm = confusion_matrix(targets_all, outputs_all.argmax(-1), labels=[0, 1, 2])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    accuracy = cm.diagonal()
    average_precision = average_precision_score(targets_one_hot, outputs_all, average=None)
    # aucroc = roc_auc_score(targets_one_hot, outputs_all, average=None)
    mean_average_precision = average_precision_score(targets_one_hot, outputs_all, average="micro")

    metric_scores = {"accuracy_ns": accuracy[0],
                     "accuracy_tss": accuracy[1],
                     "accuracy_ntss": accuracy[2],
                     "averageprecision_ns": average_precision[0],
                     "averageprecision_tss": average_precision[1],
                     "averageprecision_ntss": average_precision[2],
                     # "auroc_ns": aucroc[0],
                     # "auroc_tss": aucroc[1],
                     # "auroc_ntss": aucroc[2],
                     "mAP": mean_average_precision}

    return avg_loss, metric_scores, cm
