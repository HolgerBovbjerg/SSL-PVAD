from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram


def stack_consecutive_features(features: torch.Tensor, n_consecutive_features, stride):
    # check if padding is needed
    padding = (features.size(-1) - n_consecutive_features) % stride
    if padding:
        features = F.pad(features, (0, padding))
    features = features.unfold(dimension=-1, size=n_consecutive_features, step=stride)
    features = features.transpose(1, 2).transpose(-1, -2)
    features = features.reshape(features.size(0), features.size(1), -1).transpose(-1, -2)
    return features


class LogMelFeatureExtractor:
    def __init__(self, window_length=400, hop_length=160, n_fft=400, n_mels=40, sample_rate=16000,
                 stacked_consecutive_features: int = None, stacked_features_stride: int = 1):
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mel_spectrogram = MelSpectrogram(sample_rate=self.sample_rate,
                                              n_fft=self.n_fft,
                                              n_mels=self.n_mels,
                                              win_length=self.window_length,
                                              hop_length=self.hop_length)
        self.stacked_consecutive_features = stacked_consecutive_features
        self.stacked_features_stride = stacked_features_stride
        self.feature_rate = sample_rate / (hop_length * stacked_features_stride)

    def __call__(self, waveform: torch.Tensor):
        mel_features = self.mel_spectrogram(waveform)
        log_mel_features = torch.log10(mel_features + 1.e-6)
        if self.stacked_consecutive_features:
            log_mel_features = stack_consecutive_features(log_mel_features,
                                                          n_consecutive_features=self.stacked_consecutive_features,
                                                          stride=self.stacked_features_stride)
        return log_mel_features
