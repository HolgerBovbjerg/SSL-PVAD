from typing import Callable
import os
import time

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from APC import APC
from tqdm import tqdm

from common.misc import save_model, log


def train_single_batch(model: nn.Module, features: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor,
                       optimizer: optim.Optimizer, criterion: Callable, device: torch.device, batch_index,
                       epoch, config, scheduler):
    features, targets, lengths = features.to(device), targets.to(device), lengths.to(device)

    optimizer.zero_grad()

    predictions, targets, lengths = model(features, targets, lengths)

    loss = 0.
    for length in lengths:
        loss += criterion(predictions[:, :length], targets[:, :length])
    loss = loss / len(lengths)
    loss = loss / config["hparams"]["loss"]["accumulation_steps"]

    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["hparams"]["optimizer"]["clip_grad_norm"])

    if ((batch_index + 1) % config["hparams"]["loss"]["accumulation_steps"] == 0):
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

    return loss.item()


def train_single_epoch(model, epoch, step, train_loader, optimizer, scheduler, criterion, device, config):
    running_loss = 0.0
    epoch_start_time = time.time()

    model.train()
    batch_start_time = time.time()
    batch_step = 0
    for batch_index, data in enumerate(train_loader):
        data_load_time = time.time() - batch_start_time
        features = data[0]
        targets = data[1]
        lengths = data[2]

        ####################
        # optimization step
        ####################
        loss = train_single_batch(model=model, features=features, lengths=lengths, targets=targets,
                                  optimizer=optimizer, criterion=criterion, device=device,
                                  batch_index=batch_index, epoch=epoch, config=config,
                                  scheduler=scheduler)
        running_loss += loss
        if not step % config["exp"]["log_freq"]:
            log_dict = {"epoch": epoch,
                        "train_loss": loss,
                        "batch": batch_index,
                        "time_per_batch": time.time() - batch_start_time,
                        "data_load_time": data_load_time,
                        "lr": optimizer.param_groups[0]["lr"]}
            log(log_dict, step, config)

        step += 1
        batch_step += 1

        batch_start_time = time.time()

    log_dict = {"epoch": epoch,
                "time_per_epoch": time.time() - epoch_start_time,
                "avg_train_loss": running_loss / (batch_step)}

    return step, log_dict


def train(model: APC, optimizer: optim.Optimizer, criterion: Callable, train_loader: DataLoader,
          validation_loader: DataLoader, scheduler, config: dict):
    step = 0
    epochs = config["hparams"]["n_epochs"]
    best_avg_loss = 0.0
    device = config["exp"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

    ############################
    # start training
    ############################
    model.train()

    for epoch in tqdm(range(epochs), unit="Epoch", position=0, leave=True):
        step, log_dict = train_single_epoch(model=model,
                                            epoch=epoch,
                                            step=step,
                                            train_loader=train_loader,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            criterion=criterion,
                                            device=device,
                                            config=config)


        log(log_dict, step, config)

        if not epoch % config["exp"]["val_freq"]:
            avg_val_loss = evaluate(model, criterion, validation_loader, device)
            log_dict = {"epoch": epoch, "avg_val_loss": avg_val_loss}
            log(log_dict, step, config)

            # save best validation checkpoint
            if avg_val_loss < best_avg_loss or epoch == config["exp"]["val_freq"]:
                best_avg_loss = avg_val_loss
                save_path = os.path.join(config["exp"]["save_dir"], "best.pt")
                save_model(epoch, avg_val_loss, save_path, model, optimizer, log_file)
                save_path = os.path.join(config["exp"]["save_dir"], "best_encoder.pt")
                save_model(epoch, avg_val_loss, save_path, model.encoder, optimizer, log_file)

    ###########################
    # training complete
    ###########################

    avg_val_loss = evaluate(model, criterion, validation_loader, device)
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss}

    log(log_dict, step, config)

    # save final checkpoint
    save_path = os.path.join(config["exp"]["save_dir"], "last.pt")
    save_model(epoch, avg_val_loss, save_path, model, optimizer, log_file)
    save_path = os.path.join(config["exp"]["save_dir"], "last_encoder.pt")
    save_model(epoch, avg_val_loss, save_path, model.encoder, optimizer, log_file)

    return step


@torch.no_grad()
def evaluate(model: nn.Module, criterion: Callable, data_loader: DataLoader,
             device: torch.device) -> float:
    model.eval()

    running_loss = 0.

    for batch_index, data in (progress_bar := tqdm(enumerate(data_loader), total=len(data_loader), leave=False)):
        features = data[0].to(device)
        targets = data[1].to(device)
        lengths = data[2].to(device)

        predictions, targets, lengths = model(features, targets, lengths)
        loss = criterion(predictions, targets)

        running_loss += loss.item()
        progress_bar.set_description(f"Validation avg. loss: {running_loss / (batch_index + 1):.2f}")

        batch_step = batch_index

    model.train()

    val_avg_loss = running_loss / (batch_step + 1)

    return val_avg_loss
