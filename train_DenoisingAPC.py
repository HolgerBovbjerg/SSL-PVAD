import os
import time
from argparse import ArgumentParser
from math import ceil

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
import wandb

from common.feature_extraction import LogMelFeatureExtractor
from common.misc import count_parameters, log, seed_everything
from common.augment import get_composed_augmentations
from common.model_loader import get_model
from common.optimizer import get_optimizer
from common.scheduler import get_scheduler
from common.config_parser import get_config
from APC.noisy_trainer import evaluate, train
from APC.data_loader import pad_collate_features_clean_noisy, build_apc_datapipe
from APC.APC import DenoisingAPC


def training_pipeline(config):
    """
    Initiates and executes all the steps involved with model training and testing
    :param config: Experiment configuration
    """
    config["exp"]["save_dir"] = os.path.join(config["exp"]["exp_dir"], config["exp"]["exp_name"])
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+", encoding="utf8") as settings_file:
        settings_file.write(config_str)

    device = config["exp"]["device"]

    # feature extractor
    feature_extractor = LogMelFeatureExtractor(**config["hparams"]["audio"])

    # Augmentor
    wav_augmentor = None
    if config["hparams"]["augment"]["waveform"]:
        wav_augmentor = get_composed_augmentations(config["hparams"]["augment"]["waveform"])

    # data loaders
    train_datapipe = build_apc_datapipe(data_sets=config["data"]["train_data"],
                                        feature_extractor=feature_extractor,
                                        augmentor=wav_augmentor,
                                        load_from_tar=config["data"]["load_from_tar"],
                                        buffer_size=config["data"]["buffer_size"],
                                        batch_size=config["hparams"]["batch_size"],
                                        load_from=config["data"]["load_from"],
                                        clean_and_noisy=True,
                                        segment_max_size=config["data"]["segment_max_size"],
                                        max_token_count=config["data"]["max_token_count"],
                                        min_length=config["data"]["min_length"])
    train_loader = torch.utils.data.DataLoader(dataset=train_datapipe,
                                               batch_size=1,
                                               collate_fn=pad_collate_features_clean_noisy,
                                               num_workers=config["exp"]["n_workers"],
                                               shuffle=True)
    validation_datapipe = build_apc_datapipe(data_sets=config["data"]["validation_data"],
                                             feature_extractor=feature_extractor,
                                             augmentor=wav_augmentor,
                                             load_from_tar=config["data"]["load_from_tar"],
                                             buffer_size=config["data"]["buffer_size"],
                                             batch_size=config["hparams"]["batch_size"],
                                             clean_and_noisy=True,
                                             segment_max_size=config["data"]["segment_max_size"],
                                             max_token_count=config["data"]["max_token_count"],
                                             min_length=config["data"]["min_length"])
    validation_loader = torch.utils.data.DataLoader(dataset=validation_datapipe,
                                                    batch_size=1,
                                                    collate_fn=pad_collate_features_clean_noisy,
                                                    num_workers=config["exp"]["n_workers"],
                                                    shuffle=False)
    test_datapipe = build_apc_datapipe(data_sets=config["data"]["test_data"],
                                       feature_extractor=feature_extractor,
                                       augmentor=wav_augmentor,
                                       load_from_tar=config["data"]["load_from_tar"],
                                       buffer_size=config["data"]["buffer_size"],
                                       batch_size=config["hparams"]["batch_size"],
                                       clean_and_noisy=True,
                                       segment_max_size=config["data"]["segment_max_size"],
                                       max_token_count=config["data"]["max_token_count"],
                                       min_length=config["data"]["min_length"])
    test_loader = torch.utils.data.DataLoader(dataset=test_datapipe,
                                              batch_size=1,
                                              collate_fn=pad_collate_features_clean_noisy,
                                              num_workers=config["exp"]["n_workers"],
                                              shuffle=False)

    # create model to use as encoder in APC
    input_projection = None
    if config["hparams"]["model"]["input_projection"]:
        input_projection = nn.Linear(config["hparams"]["model"]["input_dim"],
                                     config["hparams"]["model"]["hidden_dim"])
    encoder = get_model(config["hparams"]["model"]["encoder"])

    # Create APC model
    apc = DenoisingAPC(encoder=encoder,
                       input_dim=config["hparams"]["model"]["input_dim"],
                       encoder_embedding_dim=config["hparams"]["model"]["hidden_dim"],
                       input_projection=input_projection,
                       input_dropout=config["hparams"]["model"]["input_dropout"])

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        apc.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint {args.ckpt}.")
    model = apc.to(device)
    print(f"Created model with {count_parameters(model)} parameters.")

    # Loss
    # criterion = nn.SmoothL1Loss(reduction="none", beta=config["hparams"]["loss_beta"])
    criterion = nn.L1Loss()

    # Optimizer
    optimizer = get_optimizer(model, config["hparams"]["optimizer"])

    # Learning rate scheduler
    scheduler = None
    if config["hparams"]["scheduler"]["scheduler_type"] is not None:
        if config["hparams"]["scheduler"]["steps_per_epoch"]:
            total_iters = config["hparams"]["scheduler"]["steps_per_epoch"] * max(1, (config["hparams"]["n_epochs"]))
            scheduler = get_scheduler(optimizer,
                                      scheduler_type=config["hparams"]["scheduler"]["scheduler_type"],
                                      t_max=total_iters,
                                      **config["hparams"]["scheduler"]["scheduler_kwargs"])
        else:
            total_iters = ceil(len(train_loader) / config["hparams"]["loss"]["accumulation_steps"])
            total_iters = total_iters * max(1, (config["hparams"]["n_epochs"]))

            scheduler = get_scheduler(optimizer,
                                      scheduler_type=config["hparams"]["scheduler"]["scheduler_type"],
                                      t_max=total_iters,
                                      **config["hparams"]["scheduler"]["scheduler_kwargs"])
    #####################################
    # Training Run
    #####################################
    print("Initiating training.")
    step = train(model, optimizer, criterion, train_loader, validation_loader, scheduler, config)

    #####################################
    # Final Test
    #####################################
    final_step = step + 1

    # evaluating the final state (last.pt)
    test_loss = evaluate(model, criterion, test_loader, device)
    log_dict = {
        "test_loss_last": test_loss
    }
    log(log_dict, final_step, config)

    # evaluating the best validation state (best.pt)
    ckpt = torch.load(os.path.join(config["exp"]["save_dir"], "best.pt"))
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded.")

    test_loss = evaluate(model, criterion, test_loader, device)
    log_dict = {
        "test_loss_best": test_loss
    }
    log(log_dict, final_step, config)


def main(arguments):
    """
    Calls training pipeline and sets up wandb logging if used
    """
    config = get_config(arguments.conf)
    if args.seed:
        config["hparams"]["seed"] = args.seed
    seed_everything(config["hparams"]["seed"])
    if args.id == "time":
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + "_" + time.strftime("%Y%m%d-%H%M%S")
    elif args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + "_" + args.id

    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r", encoding="utf8") as file:
                os.environ["WANDB_API_KEY"] = file.read()

        elif os.environ.get("WANDB_API_KEY", False):
            print("Found API key from env variable.")

        else:
            wandb.login()

        with wandb.init(project=config["exp"]["proj_name"],
                        name=config["exp"]["exp_name"],
                        config=config["hparams"],
                        group=config["exp"]["group_name"]):
            training_pipeline(config)

    else:
        training_pipeline(config)


if __name__ == "__main__":
    parser = ArgumentParser("Script for pretraining model with Autoregressive Predictive Coding.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint file.", default=None)
    parser.add_argument("--id", type=str, required=False, help="Additional experiment id. If 'time' is passed the "
                                                               "current time will be used", default=None)
    parser.add_argument("--seed", type=int, required=False, help="Optional random seed (overrules config file).",
                        default=None)

    args = parser.parse_args()

    main(arguments=args)
