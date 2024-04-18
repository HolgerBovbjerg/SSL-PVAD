import os
import time
from argparse import ArgumentParser
from math import ceil

import torch
import yaml

import wandb
from common import (LogMelFeatureExtractor, SpecAugment, count_parameters,
                    freeze_model_parameters, get_composed_augmentations,
                    get_loss, get_model, get_optimizer, get_scheduler,
                    load_pretrained_model, log, seed_everything,
                    wandb_log_confusion_matrix)
from common.config_parser import get_config
from PVAD import build_libriconcat_datapipe, pad_collate
from VAD import evaluate, train


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
    feature_extractor = None
    if config["data"]["load_from"] in ["raw", "decoded"]:
        feature_extractor = LogMelFeatureExtractor(**config["hparams"]["audio"])

    # Augmentor
    wav_augmentor = None
    if config["hparams"]["augment"]["waveform"]:
        wav_augmentor = get_composed_augmentations(config["hparams"]["augment"]["waveform"])

    # augmentation during training
    spec_augmentor = None
    if config["hparams"]["augment"]["spectrogram"]:
        spec_augmentor = SpecAugment(**config["hparams"]["augment"]["spectrogram"]["specaugment"])

    # data loaders
    train_datapipe = build_libriconcat_datapipe(data_sets=config["data"]["train_data"],
                                                feature_extractor=feature_extractor,
                                                augmentor=wav_augmentor,
                                                load_from_tar=config["data"]["load_from_tar"],
                                                load_from=config["data"]["load_from"],
                                                buffer_size=config["data"]["buffer_size"],
                                                waveforms_dir=config["data"]["waveforms_dir"],
                                                embeddings_dir=config["data"]["embeddings_dir"],
                                                segment_max_size=config["data"]["segment_max_size"],
                                                batch_size=config["hparams"]["batch_size"],
                                                max_token_count=config["data"]["max_token_count"],
                                                min_length=config["data"]["min_length"])
    train_loader = torch.utils.data.DataLoader(dataset=train_datapipe,
                                               batch_size=1,
                                               collate_fn=pad_collate,
                                               num_workers=config["exp"]["n_workers"],
                                               shuffle=True)
    validation_datapipe = build_libriconcat_datapipe(data_sets=config["data"]["validation_data"],
                                                     feature_extractor=feature_extractor,
                                                     augmentor=wav_augmentor,
                                                     load_from_tar=config["data"]["load_from_tar"],
                                                     load_from=config["data"]["load_from"],
                                                     buffer_size=config["data"]["buffer_size"],
                                                     waveforms_dir=config["data"]["waveforms_dir"],
                                                     embeddings_dir=config["data"]["embeddings_dir"],
                                                     segment_max_size=config["data"]["segment_max_size"],
                                                     batch_size=config["hparams"]["batch_size"],
                                                     max_token_count=config["data"]["max_token_count"],
                                                     min_length=config["data"]["min_length"])
    validation_loader = torch.utils.data.DataLoader(dataset=validation_datapipe,
                                                    batch_size=1,
                                                    collate_fn=pad_collate,
                                                    num_workers=config["exp"]["n_workers"], shuffle=False)
    test_datapipe = build_libriconcat_datapipe(data_sets=config["data"]["test_data"],
                                               feature_extractor=feature_extractor,
                                               augmentor=wav_augmentor,
                                               load_from_tar=config["data"]["load_from_tar"],
                                               load_from=config["data"]["load_from"],
                                               buffer_size=config["data"]["buffer_size"],
                                               waveforms_dir=config["data"]["waveforms_dir"],
                                               embeddings_dir=config["data"]["embeddings_dir"],
                                               segment_max_size=config["data"]["segment_max_size"],
                                               batch_size=config["hparams"]["batch_size"],
                                               max_token_count=config["data"]["max_token_count"],
                                               min_length=config["data"]["min_length"])
    test_loader = torch.utils.data.DataLoader(dataset=test_datapipe,
                                              batch_size=1,
                                              collate_fn=pad_collate,
                                              num_workers=config["exp"]["n_workers"],
                                              shuffle=False)

    # model
    model = get_model(config["hparams"]["model"]["encoder"])
    print(f"Created model with {count_parameters(model)} parameters.")

    if "checkpoint" in config["hparams"]["model"]:
        model.encoder = load_pretrained_model(model.encoder,
                                              checkpoint_path=config["hparams"]["model"]["checkpoint"][
                                              "checkpoint_path"],
                                              map_location="cpu")
        if config["hparams"]["model"]["checkpoint"]["freeze"] or config["hparams"]["model"]["checkpoint"]["freeze_first_k_epochs"]:
            freeze_model_parameters(model.encoder)
            print(f"{count_parameters(model.encoder)} parameters frozen. "
                  f"{count_parameters(model, trainable=True)} trainable parameters.")

    model = model.to(device)

    # loss
    criterion = get_loss(name=config["hparams"]["loss"]["type"])

    # optimizer
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

    step = train(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler, train_loader=train_loader,
           validation_loader=validation_loader, augmentor=spec_augmentor, config=config)

    print("Finished training.\n")

    #####################################
    # Final Test
    #####################################

    print("Starting test set evaluation.")

    final_step = step + 1
    # evaluating the final state (last.pt)
    print("Testing Last...")
    test_loss, metric_scores, conf_matrix = evaluate(model=model, criterion=criterion, data_loader=test_loader,
                                                     device=config["exp"]["device"])
    metric_scores = {f'test_{k}_last': v for k, v in metric_scores.items()}

    log_dict = {
        "test_loss_last": test_loss
    }
    log_dict.update(metric_scores)
    log(log_dict, final_step, config)

    if config["exp"]["wandb"]:
        wandb.log({"test_best_confusion_matrix": wandb_log_confusion_matrix(conf_matrix,
                                                                            class_names=["ns", "s"])},
                  step=final_step)

    # evaluating the best validation state (best.pt)
    print("Testing Best...")
    ckpt = torch.load(os.path.join(config["exp"]["save_dir"], "best.pt"))
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best checkpoint loaded...")

    test_loss, metric_scores, conf_matrix = evaluate(model=model, criterion=criterion, data_loader=test_loader,
                                                     device=config["exp"]["device"])

    metric_scores = {f'test_{k}_best': v for k, v in metric_scores.items()}

    log_dict = {
        "test_loss_best": test_loss
    }
    log_dict.update(metric_scores)
    log(log_dict, final_step, config)

    if config["exp"]["wandb"]:
        wandb.log({"test_best_confusion_matrix": wandb_log_confusion_matrix(conf_matrix,
                                                                            class_names=["ns", "s"])},
                  step=final_step)

    print("Run finished")


def main(arguments):
    """
    Calls training pipeline and sets up wandb logging if used
    """
    config = get_config(arguments.conf)
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
    parser = ArgumentParser("Training and evaluation script for VAD  model.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--id", type=str, required=False, help="Optional experiment identifier.", default=None)
    parser.add_argument("--seed", type=int, required=False, help="Optional random seed (overrules config file).",
                        default=None)

    args = parser.parse_args()

    main(arguments=args)
