import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class APC(nn.Module):
    """
    Autoregressive Predictive Coding main module.
    """

    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int,
                 encoder_embedding_dim: int,
                 input_projection: Optional[nn.Module] = None,
                 input_dropout: float = 0.):
        super().__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.encoder_embedding_dim = encoder_embedding_dim
        self.input_projection = input_projection
        self.input_dropout = nn.Dropout(input_dropout)
        self.post_network = nn.Conv1d(in_channels=encoder_embedding_dim, out_channels=input_dim,
                                      kernel_size=1, stride=1)

    def forward(self, x, lengths, time_shift: int = 3, extract_features: bool = False):
        target = x[:, time_shift:]
        x = x[:, :-time_shift]
        x = self.input_dropout(x)
        lengths = lengths - time_shift
        if self.input_projection:
            x = self.input_projection(x)
        x, _ = self.encoder(x, lengths)
        if extract_features:
            return x
        prediction = self.post_network(x.transpose(-1, -2)).transpose(-1, -2)
        return prediction, target, lengths


class DenoisingAPC(nn.Module):
    """
    Denoising Autoregressive Predictive Coding main module.
    """

    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int,
                 encoder_embedding_dim: int,
                 input_projection: Optional[nn.Module] = None,
                 input_dropout: float = 0.):
        super().__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.encoder_embedding_dim = encoder_embedding_dim
        self.input_projection = input_projection
        self.input_dropout = nn.Dropout(input_dropout)
        self.post_network = nn.Conv1d(in_channels=encoder_embedding_dim, out_channels=input_dim,
                                      kernel_size=1, stride=1)

    def forward(self, x, target, lengths, time_shift: int = 3, extract_features: bool = False):
        target = target[:, time_shift:]
        x = x[:, :-time_shift]
        x = self.input_dropout(x)
        lengths = lengths - time_shift
        if self.input_projection:
            x = self.input_projection(x)
        x, hidden = self.encoder(x, lengths)
        if extract_features:
            return x
        prediction = self.post_network(x.transpose(-1, -2)).transpose(-1, -2)
        return prediction, target, lengths
