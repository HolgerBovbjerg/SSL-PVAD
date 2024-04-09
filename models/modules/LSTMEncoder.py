from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0., projection_size=0, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
                            proj_size=projection_size, **kwargs)

    def forward(self, x, lengths, hidden=None, output_hidden: bool = False):
        # first pack the padded sequences
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # lstm pass
        out_packed, hidden = self.lstm(x_packed, hidden)
        out_padded, lengths = pad_packed_sequence(out_packed, batch_first=True)
        output = (out_padded, lengths)
        if output_hidden:
            output += (hidden,)

        return output


class LSTMEncoder2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=None):
        super().__init__()
        in_sizes = ([input_dim] +
                    [hidden_dim] * (num_layers - 1))
        out_sizes = [hidden_dim] * num_layers
        self.rnns = nn.ModuleList(
            [nn.LSTM(input_size=in_size, hidden_size=out_size, batch_first=True)
             for (in_size, out_size) in zip(in_sizes, out_sizes)])
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths, hidden=None, output_hidden_states: bool = False, mask: Optional[torch.Tensor] = None):
        # Mask input
        if mask is not None:
            x[mask] = 0.
        # first pack the padded sequences
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # lstm pass
        out_padded = 0
        hidden_states = []
        out_packed = x_packed
        for i, layer in enumerate(self.rnns):
            out_packed, _ = layer(out_packed, hidden)
            out_padded, lengths = pad_packed_sequence(out_packed, batch_first=True)
            # outputs: (batch_size, seq_len, rnn_hidden_size)
            if hasattr(self, 'dropout') and (i + 1 < len(self.rnns)):
                # apply dropout except the last rnn layer
                out_padded = self.dropout(out_padded)
            hidden_states.append(out_padded)

        output = (out_padded, lengths)
        if output_hidden_states:
            output += (hidden_states,)
        return output
