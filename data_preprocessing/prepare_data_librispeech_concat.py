from argparse import ArgumentParser
import math
from pathlib import Path
from glob import glob
import random
import re
import yaml

import pandas as pd
import numpy as np
import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm

from common.augment import AddRIR, AddNoise
from common.config_parser import get_config
from common.misc import seed_everything
from common.feature_extraction import LogMelFeatureExtractor
from common.speaker_embedding import generate_embeddings


def interpolate_labels(timestamps_interp, label_timestamps, labels):
    previous_index = 0
    labels_interp = np.zeros_like(timestamps_interp)
    for label, timestamp in zip(labels, label_timestamps):
        index = np.searchsorted(timestamps_interp, timestamp)
        labels_interp[previous_index:index] = label
        previous_index = index
    return labels_interp


def trim_utterance_end(x, sample_rate, time_stamps):
    """Trim the end of the utterance.

    Trim the end of the utterance so that the alignment timestamps for the other
    utterances can be offset by exactly n frames and so that the utterance's length
    is divisible by 10ms.

    Args:
        x (np.array): The source waveform to be trimmed.
        sample_rate (int): Sample rate of x.
        time_stamps (list of strings): The time stamp array corresponding to x.
    """
    signal_length = x.size(-1)
    end_stamp = math.trunc(time_stamps[-1] * 100) / 100
    end = end_stamp * sample_rate
    if signal_length != end:
        assert (signal_length > end), "Signal length was smaller than the end timestamp"
        x = x[:, :int(end - signal_length)]

    return x, end_stamp


def read_unaligned(file_path):
    with open(file_path, encoding="utf8") as file:
        lines = file.readlines()
        unaligned_identifiers = [line.split(" ")[0] for line in lines if "#" not in line]
    return unaligned_identifiers


def read_alignment(alignment_root, identifiers):
    speaker = str(identifiers[0])
    chapter = str(identifiers[1])
    utterance = str(identifiers[2])
    # Make sure utterance id is four digits
    utterance = f"{utterance:0>4}"

    alignment_identifier = '-'.join((speaker, chapter, utterance))

    path = alignment_root + speaker + "/" + chapter + "/"
    alignment_file = glob(path + "*alignment.txt")[0]
    with open(alignment_file, encoding="utf8") as file:
        timestamps = None
        for line in file.readlines():
            if alignment_identifier in line:
                aligned_text = line.split(' ')[1][1:-1]
                timestamps = line.split(' ')[2][1:-2]  # remove newline
                timestamps = timestamps.split(',')
                timestamps = np.array([float(timestamp) for timestamp in timestamps])
                labels = re.sub(r"[A-Z']+", 'W', aligned_text)  # we only need word or no word ("W", "")
                labels = np.char.array(labels.split(","), itemsize=4)
                break

        if timestamps is None:
            print(f"No alignment found for {alignment_identifier}")

    return timestamps, labels


def generate_vad_labels(utterance: torch.Tensor):
    return torch.ones(utterance.size())


def main(arguments):
    with open(arguments.conf, "r", encoding="utf8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    files_per_dir = arguments.files_per_dir

    librispeech_root = config["data"]["librispeech_root"]
    alignments_root = config["data"]["alignments_root"]

    data_root = config["data"]["data_root"]
    waveforms_dir = config["data"]["waveforms_dir"]
    embeddings_dir = config["data"]["embeddings_dir"]
    features_dir = config["data"]["features_dir"]

    if args.splits:
        splits = args.splits
    else:
        splits = config["data"]["train_splits"] + config["data"]["validation_splits"] + config["data"]["test_splits"]

    sampling_rate = config["hparams"]["audio"]["sample_rate"]
    hop_length = config["hparams"]["audio"]["hop_length"]
    stacked_consecutive_features = config["hparams"]["audio"]["stacked_consecutive_features"]
    stacked_features_stride = config["hparams"]["audio"]["stacked_features_stride"]
    if stacked_features_stride is None:
        stacked_features_stride = 1

    feature_extractor = LogMelFeatureExtractor(**config["hparams"]["audio"])

    for split in splits:
        seed_everything(config["hparams"]["seed"])
        print(f"Generating concatenated utterances for {split}...")

        metadata = []
        metadata_columns = ["identifier", "subdir", "sampling_rate", "target_speaker_id", "target_speaker_chapter_id",
                            "target_speaker_utterance_id", "duration"]
        metadata_path = data_root + split + "/metadata.csv"

        root = Path(librispeech_root).parent
        data = LIBRISPEECH(root=root,
                           url=split,
                           download=True,
                           folder_in_archive="LibriSpeech")

        if arguments.generate_embeddings:
            print(f"Generating speaker embeddings for speakers in {split}...")
            generate_embeddings(libri_root=librispeech_root, split=split,
                                save_dir=data_root + split + "/" + embeddings_dir)

        if arguments.generate_utterances:
            # Remove unaligned utterances from data_indexes
            unaligned_identifiers = read_unaligned(alignments_root + "unaligned.txt")
            data_indexes = list(range(len(data)))
            for index, identifier in enumerate(data._walker):
                if identifier in unaligned_identifiers:
                    data_indexes.remove(index)

            # Generating N concatenated utterances (or until dataset is exhausted)
            waveforms_current_dir = ""
            features_current_dir = ""
            subdir = 0
            for iteration in tqdm(range(arguments.N), unit=" sample"):
                # Sample utterances
                n_utterances = random.randint(1, 3)
                try:
                    utterance_indexes = random.sample(data_indexes, k=n_utterances)
                except ValueError:
                    print(f"Ran out of utterances in {split}. Ending feature generation for {split}.")
                    break

                utterances = [data[index] for index in utterance_indexes]

                # Sanity checks
                if any([utterance[0].size(1) == 0 for utterance in utterances]):
                    print("Found zero length utterance, skipping...")
                    continue

                identifiers = []
                for utterance in utterances:
                    speaker, chapter, segment = map(str, utterance[3:])
                    segment = f"{segment:0>4}"
                    identifiers.append("-".join([speaker, chapter, segment]))

                unaligned = [identifier in unaligned_identifiers for identifier in identifiers]
                if any(unaligned):
                    print("Found unaligned utterance, skipping...")
                    continue

                # Check sampling rate of utterances
                assert all(utterance[1] == sampling_rate for utterance in utterances), \
                    f"Sampling rate differs from {sampling_rate}"

                # Randomly select target speaker
                target_speaker_index = random.randint(0, n_utterances - 1)
                target_speaker_utterance = utterances[target_speaker_index]
                target_speaker_id, target_speaker_chapter_id, target_speaker_utterance_id = target_speaker_utterance[3:]

                if args.unique:  # All utterances used only once
                    for index in utterance_indexes:
                        data_indexes.remove(index)
                elif args.unique_target:  # Only target speaker utterance is unique
                    data_indexes.remove(utterance_indexes[target_speaker_index])

                # Read timestamps from word alignment files
                label_timestamps, labels = zip(
                    *[read_alignment(alignments_root + split + "/", utterance[3:]) for utterance in utterances])

                for i, label in enumerate(labels):
                    label[label == ""] = "ns"
                    if i == target_speaker_index:
                        label[label == 'W'] = "tss"
                    else:
                        label[label == 'W'] = "ntss"

                label_timestamps = list(label_timestamps)
                trimmed_utterances, end_stamps = zip(
                    *[trim_utterance_end(utterance[0], utterance[1], label_timestamps[i])
                      for i, utterance in enumerate(utterances)])
                for i in range(1, len(end_stamps)):
                    label_timestamps[i] = label_timestamps[i] + sum(end_stamps[0:i])

                if any(utterance.size(1) == 0 for utterance in trimmed_utterances):
                    print("Utterance length zero after trimming, skipping...")
                    continue

                # Concatenate utterances and label information
                concatenated_utterances = torch.cat(trimmed_utterances, dim=-1)
                concatenated_label_timestamps = np.concatenate(label_timestamps)
                concatenated_labels = np.concatenate(labels)

                # Add RIR and noise if specified
                if "augment" in config["hparams"]:
                    rir_augmentor = AddRIR(**config["hparams"]["augment"]["waveform"]["rir"])
                    noise_augmentor = AddNoise(**config["hparams"]["augment"]["waveform"]["noise"])

                    if args.reverberate:
                        concatenated_utterances = rir_augmentor(concatenated_utterances)
                    if args.add_noise:
                        concatenated_utterances = noise_augmentor(concatenated_utterances)

                duration = len(concatenated_utterances) / sampling_rate

                # Generate VAD labels from timestamps
                label2number = config["data"]["labels"]
                concatenated_labels_numeric = np.empty_like(concatenated_labels, dtype=int)
                for i, label in enumerate(concatenated_labels):
                    concatenated_labels_numeric[i] = label2number[label]

                # Test if new subdirectory should be created
                if iteration % files_per_dir == 0:
                    subdir = str(iteration // files_per_dir)
                    waveforms_current_dir = data_root + split + "/" + waveforms_dir + subdir + "/"
                    Path(waveforms_current_dir).mkdir(parents=True, exist_ok=False)


                # Save data
                identifier = "_".join([str(utterance[3]) + "-" + str(utterance[4]) + "-" + str(utterance[5])
                                       for utterance in utterances])
                metadata.append([identifier, subdir, sampling_rate, target_speaker_id, target_speaker_chapter_id,
                                 target_speaker_utterance_id, duration])

                waveform_save_path = waveforms_current_dir + identifier
                torchaudio.save(waveform_save_path + ".flac", concatenated_utterances, sampling_rate)
                torch.save([concatenated_label_timestamps, concatenated_labels_numeric], f=waveform_save_path + "_labels.pt")

                if arguments.extract_features:
                    features = feature_extractor(concatenated_utterances)

                    feature_period = hop_length * stacked_consecutive_features * stacked_features_stride / sampling_rate
                    timestamp_features = np.round(np.arange(features.size(-1)) * feature_period, 2)

                    features_vad_labels = interpolate_labels(timestamps_interp=timestamp_features,
                                                             label_timestamps=concatenated_label_timestamps,
                                                             labels=concatenated_labels_numeric)
                    features_vad_labels = torch.from_numpy(features_vad_labels)

                    feature_save_path = features_current_dir + identifier + ".pt"
                    torch.save((features, features_vad_labels), f=feature_save_path)

                    # Test if new subdirectory should be created
                    if iteration % files_per_dir == 0:
                        features_current_dir = data_root + split + "/" + features_dir + subdir + "/"
                        Path(features_current_dir).mkdir(parents=True, exist_ok=False)

            # Save metadata
            print("Saving metadata...")
            metadata_df = pd.DataFrame(metadata, columns=metadata_columns)
            metadata_df.to_csv(metadata_path)

        print(f"Finished processing {split = }")

    print("Data preparation finished.")


if __name__ == "__main__":
    parser = ArgumentParser("Data preparation script for PVAD LibriSpeech Concatenated data set.")
    parser.add_argument("--conf", type=str, required=True, help="Path to .yaml file with configuration.")
    parser.add_argument("--files_per_dir", type=int, default=2000,
                        help="Features are split into subdirectories of this size (in data samples).")
    parser.add_argument("--generate_utterances", action='store_true',
                        help="Whether to generate concatenated_utterances.")
    parser.add_argument("--generate_embeddings", action='store_true',
                        help="Whether to generate speaker embeddings.")
    parser.add_argument("--extract_features", action='store_true',
                        help="Whether to extract from waveform and save logMel features.")
    parser.add_argument("--unique", action='store_true',
                        help="Only use utterances once.")
    parser.add_argument("--unique_target", action='store_true',
                        help="Only use utterances as target speaker once.")
    parser.add_argument("--N", type=int, default=300000, help="Number of concatenated utterances to generate")
    parser.add_argument("--splits", nargs='+', type=str, default=None,
                        help="Name of LibriSpeech splits to preprocess")
    args = parser.parse_args()

    main(arguments=args)
