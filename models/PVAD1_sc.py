from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PVAD1_SC(nn.Module):
    """Personal VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers=3, out_dim=2):
        """PersonalVAD class initializer.
        Args:
            input_dim (int): Input feature vector size.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers in the model.
            out_dim (int): Number of neurons in the output layer.
        """

        super(PVAD1_SC, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        # define the model layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

        # similarity score scaling parameters
        self.similarity_weight = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor(-5.0), requires_grad=True)

    def forward(self, x, similarity_scores, x_lens, hidden=None, output_hidden=None):
        """Personal VAD model forward pass method."""
        x_packed = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.lstm(x_packed, hidden)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)

        out_padded = self.fc(out_padded)

        similarity_scores = similarity_scores * self.similarity_weight + self.similarity_bias
        output = torch.stack([out_padded[:, :, 0],
                              out_padded[:, :, 1] * similarity_scores,
                              out_padded[:, :, 1] * (1 - similarity_scores)],
                             dim=-1)
        output = (output, x_lens)
        if output_hidden:
            output += (hidden,)
        return output


if __name__ == "__main__":
    model = PVAD1_SC(input_dim=40, hidden_dim=64, num_layers=2)

    features = torch.ones(32, 100, 40)
    lengths = torch.ones(32, ) * 100
    similarity = torch.ones(32, 100)

    out = model(x=features, similarity_scores=similarity, x_lens=lengths)

    print("done")
