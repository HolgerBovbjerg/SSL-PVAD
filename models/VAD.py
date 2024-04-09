import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMVAD(nn.Module):
    """VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers=2, out_dim=2):
        """VAD class initializer.
        Args:
            input_dim (int): Input feature vector size.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers in the model.
            out_dim (int): Number of neurons in the output layer.
        """

        super(LSTMVAD, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        # define the model encoder...
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # use the original PersonalVAD configuration with one additional layer
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x,lengths, hidden=None, output_hidden=None):
        """VAD model forward pass method."""

        # Pass through lstm
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.encoder(x_packed, hidden)
        out_padded, lengths = pad_packed_sequence(out_packed, batch_first=True)

        # Project to output dimensionality
        out_padded = self.fc(out_padded)

        output = (out_padded, lengths)
        if output_hidden:
            output += (hidden,)
        return output
