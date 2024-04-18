from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata import datapipes as dp

from common.misc import count_files
from data_preprocessing.prepare_data_librispeech_concat import \
    interpolate_labels

implemented_data_sets = ["librispeech_concat"]

librispeech_splits = ["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100", "train-clean-360",
                      "train-other-500"]


def filter_fn(filename, file_types):
    return filename.endswith(file_types)


def decode(item):
    key, value = item

    text_file_ending = (".txt", ".bin", ".sr", ".hop_length", ".window_length", ".n_fft", ".n_mels")

    if key.endswith(text_file_ending):
        return key, value.read().decode("utf-8")
    else:
        return key, value.read()

def is_audio_path(item):
    a = item.endswith(".wav") or item.endswith(".flac")
    b = not "_" in item.split(".")[-2]
    return a and b  # and not "_" in item.split(".")[-2]


def to_item(wds_item, load_from: str):
    if load_from == "raw":
        if ".flac" in wds_item.keys():
            audio_ext = ".flac"
        else:
            audio_ext = ".wav"
        return wds_item["__key__"], wds_item[audio_ext], wds_item[".labels.pth"]
    elif load_from == "decoded":
        return wds_item["__key__"], wds_item[".pth"], wds_item[".labels.pth"], wds_item[".sr"]
    return wds_item["__key__"], wds_item[".pth"], wds_item[".labels.pth"], wds_item[".sr"], wds_item[".hop_length"]


def add_label_path(item):
    audio_path = item
    labels_path = str(Path(audio_path).parent / Path(audio_path).stem) + "_labels.pt"
    return labels_path


def to_wds_style(item):
    audio_path, audio_filestream = item[0]
    labels_path, labels_filestream = item[1]
    return audio_path, audio_filestream, labels_filestream


def load_raw_waveform(item):
    audio_path, waveform_file_stream, labels_filestream = item
    with BytesIO(waveform_file_stream) as fd:
        waveform, sampling_rate = torchaudio.load(fd)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1.e-9)
    return waveform, sampling_rate, labels_filestream, audio_path


def load_decoded_waveform(item):
    audio_path, waveform_file_stream, labels_filestream, sampling_rate = item
    with BytesIO(waveform_file_stream) as fd:
        waveform = torch.load(fd)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1.e-9)
    return waveform, sampling_rate, labels_filestream, audio_path


def load_features(item):
    feature_path, feature_file_stream, labels_filestream, sampling_rate, hop_length = item
    with BytesIO(feature_file_stream) as fd:
        features = torch.load(fd)
    features.requires_grad = True
    sampling_rate = round(sampling_rate / hop_length)
    return features, sampling_rate, labels_filestream, feature_path


def augment(item, augmentor):
    waveform, sampling_rate, labels_filestream, audio_path = item
    return augmentor(waveform.unsqueeze(0))[0], sampling_rate, labels_filestream, audio_path


def extract_features(item, feature_extractor):
    waveform, sampling_rate, labels_filestream, audio_path = item
    features = feature_extractor(waveform)
    sampling_rate = round(sampling_rate / feature_extractor.hop_length)
    return features, sampling_rate, labels_filestream, audio_path


def load_labels(item):
    features, sampling_rate, labels_filestream, audio_path = item
    with BytesIO(labels_filestream) as fd:
        concatenated_label_timestamps, concatenated_labels_numeric = torch.load(fd)
    timestamp_features = (np.arange(features.size(-1)) / sampling_rate)
    labels = interpolate_labels(timestamps_interp=timestamp_features,
                                label_timestamps=concatenated_label_timestamps,
                                labels=concatenated_labels_numeric)
    labels = torch.from_numpy(labels).to(torch.int64)
    return features, labels, sampling_rate, audio_path


def load_speaker_embedding(item, metadata: pd.DataFrame, speaker_embeddings: dict):
    features, labels, sampling_rate, audio_path = item
    identifier = Path(audio_path).stem
    target_speaker_id = int(metadata.loc[identifier]["target_speaker_id"])
    speaker_embedding = speaker_embeddings[str(target_speaker_id)]
    return features, labels, speaker_embedding, sampling_rate, target_speaker_id, audio_path


def segment_features(item, segment_size, drop_last: bool = False,
                     min_length: int = 0):
    features, labels, speaker_embedding, sampling_rate, target_speaker_id, audio_path = item
    features = features.split(segment_size, dim=-1)
    labels = labels.split(segment_size, dim=-1)
    if (drop_last or (len(features[-1]) < min_length) and (len(features) > 1)):
        features = features[:-1]
        labels = labels[:-1]

    output = [(feature_segment, label_segment, speaker_embedding, target_speaker_id)
              for feature_segment, label_segment in zip(features, labels)]

    return output

def pad_collate(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    features = [data[0][0].T for data in batch[0]]
    features = [(feature - feature.mean()) / (feature.std() + 1.e-9)
                for feature in features]  # Instance norm
    labels = [data[1] for data in batch[0]]
    speaker_embeddings = torch.stack([data[2] for data in batch[0]])
    target_speaker_ids = [data[3] for data in batch[0]]

    lengths = torch.tensor([feature.size(0) for feature in features])

    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    return features_padded, labels_padded, speaker_embeddings, lengths, target_speaker_ids


def pad_collate_features(batch):
    """Padding function used to deal with batches of sequences of variable lengths."""
    features = [data[0][0].T for data in batch]
    features = [(feature - feature.mean()) / (feature.std() + 1.e-9)
                for feature in features]  # Instance norm
    lengths = torch.tensor([feature.size(0) for feature in features])

    features_padded = pad_sequence(features, batch_first=True, padding_value=0)

    return features_padded, lengths


def len_fn(item):
    return item[0].size(-1)


def build_libriconcat_datapipe(data_sets, feature_extractor, waveforms_dir: str = "Waveforms/",
                               metadata_filename: str = "metadata.csv", embeddings_dir: str = "SpeakerEmbeddings/",
                               embeddings_filename: str = "speaker_embeddings.pt", features_only: bool = False,
                               augmentor: Callable = None, load_from_tar: bool = False, buffer_size: int = 10000,
                               load_from: str = "raw", batch_size: int = 1,
                               segment_max_size: int = None, min_length: int = 0, max_token_count: int = None):
    if load_from in ["decoded", "features"]:
        file_types = (".pth", ".pt")
    elif load_from == "raw":
        file_types = (".wav", '.flac')
    else:
        file_types = ()
        assert ValueError(f"Loading_method: {load_from} not supported. "
                          "Supported methods are 'raw', 'decoded' and 'features'")

    length = 0

    # List all files
    datapipes = []
    metadata = []
    speaker_embeddings = {}
    for data_set_name, info in data_sets.items():
        for split in info["splits"]:
            if load_from_tar:
                datapipe = dp.iter.FileLister(info["root"] + split, "*.tar")
            else:
                datapipe = dp.iter.FileLister(info["root"] + split, recursive=True,
                                              masks=[f"*{file_type}" for file_type in file_types])
            datapipes.append(datapipe)
            metadata.append(pd.read_csv(info["root"] + split + "/" + metadata_filename).set_index('identifier'))
            speaker_embeddings.update(
                torch.load(info["root"] + split + "/" + embeddings_dir + "/" + embeddings_filename))
            length += count_files(info["root"] + split + "/" + waveforms_dir, ".flac")
    metadata = pd.concat(metadata)

    # Concatenate filelists
    datapipe = dp.iter.Concater(*datapipes)

    # Shuffle files and apply sharding filter
    datapipe = datapipe.shuffle(buffer_size=buffer_size).sharding_filter()

    # Open files
    if load_from_tar:
        datapipe = dp.iter.FileOpener(datapipe, mode="b")
        datapipe = datapipe.load_from_tar().map(decode).webdataset().map(partial(to_item, load_from=load_from))
    else:
        datapipe, datapipe_labels = datapipe.fork(num_instances=2)
        datapipe_labels = datapipe_labels.map(add_label_path)
        datapipe = dp.iter.FileOpener(datapipe, mode="b")
        datapipe_labels = dp.iter.FileOpener(datapipe_labels, mode="b")
        datapipe = datapipe.map(decode)
        datapipe_labels = datapipe_labels.map(decode)
        datapipe = datapipe.zip(datapipe_labels)
        datapipe = datapipe.map(to_wds_style)

    # Load features or generate from waveforms
    if load_from == "features":
        datapipe = datapipe.map(load_features)
    else:
        if load_from == "decoded":
            datapipe = datapipe.map(load_decoded_waveform)
        else:
            datapipe = datapipe.map(load_raw_waveform)
        if augmentor:
            datapipe = datapipe.map(partial(augment, augmentor=augmentor))
        datapipe = datapipe.map(partial(extract_features, feature_extractor=feature_extractor))

    datapipe = datapipe.map(load_labels)
    datapipe = datapipe.map(partial(load_speaker_embedding,
                                    metadata=metadata,
                                    speaker_embeddings=speaker_embeddings))
    if segment_max_size:
        datapipe = datapipe.flatmap(
            partial(segment_features, segment_size=segment_max_size, min_length=min_length))
        datapipe = datapipe.shuffle(buffer_size=1000)
    if max_token_count:
        datapipe = datapipe.max_token_bucketize(max_token_count=max_token_count, len_fn=len_fn, include_padding=True,
                                                buffer_size=100, min_len=min_length)
    else:
        datapipe = datapipe.batch(batch_size=batch_size, drop_last=True)
    datapipe = datapipe.set_length(length)
    return datapipe


def get_data_loader(dataset, batch_size: int = 1, num_workers: int = 0, shuffle=True, collate_fn=pad_collate,
                    pin_memory: bool = False):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)
    return data_loader
