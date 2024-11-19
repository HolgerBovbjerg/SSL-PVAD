from copy import deepcopy
from pathlib import Path
from argparse import ArgumentParser
import yaml

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score
import pandas as pd

from common.model_loader import get_model
from common.feature_extraction import LogMelFeatureExtractor
from common.augment import get_augmentor
from common.model_loader import load_pretrained_model
from common.misc import seed_everything
from PVAD.data_loader import build_libriconcat_datapipe, pad_collate


@torch.no_grad()
def evaluate(model, data_loader, device):
    seed_everything(42)

    outputs_list = []
    targets_list = []

    model = model.to(device)
    model.eval()
    for data in tqdm(data_loader, mininterval=10.0):
        features = data[0].to(device)
        targets = data[1].to(device)
        speaker_embeddings = data[2].to(device)
        lengths = data[3].to(device)

        predictions, lengths = model(features, speaker_embeddings, lengths)

        for j in range(predictions.size(0)):
            probabilities = torch.nn.functional.softmax(predictions[j][:lengths[j]], dim=-1)
            outputs_list.append(probabilities.cpu().numpy())
            targets_list.append(targets[j][:lengths[j]].cpu().numpy())

    targets_all = np.concatenate(targets_list)
    outputs_all = np.concatenate(outputs_list)
    targets_one_hot = np.eye(3)[targets_all]

    conf_matrix = confusion_matrix(targets_all, outputs_all.argmax(axis=-1), labels=[0, 1, 2])
    conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    accuracy = np.diag(conf_matrix_normalized)
    aucroc = roc_auc_score(targets_one_hot, outputs_all, average=None, labels=[0, 1, 2])
    average_precision = average_precision_score(targets_one_hot, outputs_all, average=None)
    return accuracy, aucroc, average_precision


def main(args):
    data_root = args.data_root
    noise_root = args.noise_root
    model_name = Path(args.checkpoint_path).parent.stem
    checkpoint_path = Path(args.checkpoint_path)
    save_folder = args.save_folder
    save_folder = save_folder + model_name + "/"
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    splits = ["test-clean"]
    data_sets = {"librispeechconcat": {"root": data_root, "splits": splits}}

    # Features extractor
    feature_config = {'sample_rate': 16000,
                      'n_mels': 40,
                      'n_fft': 400,
                      'window_length': 400,
                      'hop_length': 160}
    feature_extractor = LogMelFeatureExtractor(**feature_config)

    # test device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # test settings
    SNRs = [-5, 0, 5, 10, 15, 20, "clean"]
    noise_types = ["bbl", "bus", "caf", "ssn", "str", "ped"]

    # testing
    scores = {}
    labels = ["ns", "ts", "nts"]
    # Check
    save_path = save_folder + "/test_scores_" + model_name + ".csv"
    if Path(save_path).exists():
        #print(f"Model evaluation file for {model_name=} already exists.")
        raise ValueError(f"Model evaluation file for {model_name=} already exists.")

    # Get model config
    yaml.add_multi_constructor('tag:yaml.org,2002:python/object/apply:torch.device',
                                lambda loader, suffix, node: None)
    with open(str(checkpoint_path.parent / "settings.txt"), "r", encoding="utf8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    model_config = config["hparams"]["model"]["encoder"]

    # Load model architecture and pretrained weights
    model = get_model(model_config)
    test_model = load_pretrained_model(deepcopy(model), checkpoint_path=str(checkpoint_path), map_location="cpu")

    # Start evaluation of model
    scores[model_name] = {noise_type: {} for noise_type in noise_types}
    print(f"Evaluating {model_name}:")

    for SNR in SNRs:
        print(f"Evaluating at {SNR=}.")
        if SNR == "clean":
            augmentor = None
            test_datapipe = build_libriconcat_datapipe(data_sets=data_sets,
                                                        feature_extractor=feature_extractor,
                                                        augmentor=augmentor,
                                                        load_from_tar=True,
                                                        load_from="raw",
                                                        segment_max_size=None,
                                                        batch_size=1)
            test_loader = torch.utils.data.DataLoader(dataset=test_datapipe,
                                                        batch_size=1,
                                                        collate_fn=pad_collate,
                                                        num_workers=16,
                                                        shuffle=False)
            accuracy, aucroc, average_precision = evaluate(model=test_model,
                                                            data_loader=test_loader,
                                                            device=device)
            scores[model_name]["clean"] = {"accuracy": {label: accuracy[i]
                                                        for i, label in enumerate(labels)},
                                            "aucroc": {label: aucroc[i]
                                                        for i, label in enumerate(labels)},
                                            "average_precision": {label: average_precision[i]
                                                                    for i, label in enumerate(labels)}
                                            }
        else:
            for noise_type in noise_types:
                print(f"{noise_type=}.")

                augmentor = get_augmentor(name="noise",
                                            noise_paths=noise_root + "/" + noise_type + "/",
                                            snr_db_min=SNR,
                                            snr_db_max=SNR)

                test_datapipe = build_libriconcat_datapipe(data_sets=data_sets,
                                                        feature_extractor=feature_extractor,
                                                        augmentor=augmentor,
                                                        load_from_tar=True,
                                                        load_from="raw",
                                                        segment_max_size=0,
                                                        batch_size=1)
                test_loader = torch.utils.data.DataLoader(dataset=test_datapipe,
                                                        batch_size=1,
                                                        collate_fn=pad_collate,
                                                        num_workers=16,
                                                        shuffle=False)

                accuracy, aucroc, average_precision = evaluate(model=test_model,
                                                            data_loader=test_loader,
                                                            device=device)
                scores[model_name][noise_type][SNR] = {"accuracy": {label: accuracy[i]
                                                                        for i, label in enumerate(labels)},
                                                            "aucroc": {label: aucroc[i]
                                                                    for i, label in enumerate(labels)},
                                                            "average_precision": {label: average_precision[i] 
                                                                                for i, label in enumerate(labels)}
                                                        }
                                                

    # Flatten the dictionary
    flat_data = []
    for noise, noise_levels in scores[model_name].items():
        if noise == "clean":
            metrics = noise_levels
            for metric, labels in metrics.items():
                for label, value in labels.items():
                    row = {'model': model_name, 'SNR': noise, 'noise_type': noise, "metric": metric, "label": label, "score": value}
                    flat_data.append(row)
        else:
            for noise_level, metrics in noise_levels.items():
                for metric, labels in metrics.items():
                    for label, value in labels.items():
                        row = {'model': model_name, 'SNR': noise_level, 'noise_type': noise, "metric": metric, "label": label, "score": value}
                        flat_data.append(row)

    test_results_df = pd.DataFrame(flat_data)
    test_results_df.to_csv(save_path, sep=",", index=False, encoding="utf-8")
    

if __name__ == "__main__":
    parser = ArgumentParser("Evaluation script for PVAD model.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to folder with data.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to folder with checkpoints.")
    parser.add_argument("--noise_root", type=str, required=True, help="Path to folder with noise.")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to folder where results should be saved.")

    args = parser.parse_args()

    main(args=args)
