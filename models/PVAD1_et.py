import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.modules.modulation import FiLM


class PVAD1ET(nn.Module):
    """Personal VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers=2, out_dim=3, embedding_dim=256):
        """PersonalVAD class initializer.
        Args:
            input_dim (int): Input feature vector size.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers in the model.
            out_dim (int): Number of neurons in the output layer.
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim

        # define the model encoder...
        self.encoder = nn.LSTM(input_dim + embedding_dim, hidden_dim, num_layers, batch_first=True)

        # use the original PersonalVAD configuration with one additional layer
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, speaker_embeddings, lengths, hidden=None, output_hidden=None):
        """Personal VAD model forward pass method."""
        # Concatenate features and speaker embedding
        speaker_embeddings = speaker_embeddings.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, speaker_embeddings], dim=-1)

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


class PVAD1ET2(nn.Module):
    """Personal VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers=2, out_dim=3, embedding_dim=256):
        """PersonalVAD class initializer.
        Args:
            input_dim (int): Input feature vector size.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers in the model.
            out_dim (int): Number of neurons in the output layer.
            use_fc (bool, optional): Specifies, whether the model should use the
                last fully-connected hidden layer. Defaults to True.
            linear (bool, optional): Specifies the activation function used by the last
                hidden layer. If False, the tanh is used, if True, no activation is
                used. Defaults to False.
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim

        # LSTM encoder
        self.speaker_embedding_encoder = nn.Linear(embedding_dim, input_dim)
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Linear classification head
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, speaker_embeddings, lengths, hidden=None, output_hidden=None):
        """Personal VAD model forward pass method."""

        # Project and add speaker embeddings
        speaker_embeddings = speaker_embeddings.unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + self.speaker_embedding_encoder(speaker_embeddings)

        # Pass through lstm
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.encoder(x_packed, hidden)
        out_padded, lengths = pad_packed_sequence(out_packed, batch_first=True)

        # Classify
        out_padded = self.fc(out_padded)

        output = (out_padded, lengths)
        if output_hidden:
            output += (hidden,)
        return output


class PVAD1ET22(nn.Module):
    """Personal VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers=2, out_dim=3, embedding_dim=256):
        """PersonalVAD class initializer.
        Args:
            input_dim (int): Input feature vector size.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers in the model.
            out_dim (int): Number of neurons in the output layer.
            use_fc (bool, optional): Specifies, whether the model should use the
                last fully-connected hidden layer. Defaults to True.
            linear (bool, optional): Specifies the activation function used by the last
                hidden layer. If False, the tanh is used, if True, no activation is
                used. Defaults to False.
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim

        # LSTM encoder
        self.speaker_embedding_encoder = nn.Linear(embedding_dim, input_dim)
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Linear classification head
        self.fc = nn.Linear(hidden_dim, out_dim)

        # Similarity score scaling parameters
        self.similarity_weight = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor(-5.0), requires_grad=True)

    def forward(self, x, speaker_embeddings, lengths, hidden=None, output_hidden=None):
        """Personal VAD model forward pass method."""

        # Project and add speaker embeddings
        speaker_embeddings = speaker_embeddings.unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + self.speaker_embedding_encoder(speaker_embeddings)

        # Pass through lstm
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.encoder(x_packed, hidden)
        out_padded, lengths = pad_packed_sequence(out_packed, batch_first=True)

        # Linear
        out_padded = self.fc(out_padded)

        # Explicitly divide into VAD and similarity score
        vad_padded = out_padded[:, :, :-1]
        similarity_scores_padded = out_padded[:, :, -1]
        similarity_scores_padded = similarity_scores_padded * self.similarity_weight + self.similarity_bias

        # Combine similarity score and VAD
        output_padded = torch.stack([vad_padded[:, :, 0],
                                     vad_padded[:, :, 1] * similarity_scores_padded,
                                     vad_padded[:, :, 1] * (1 - similarity_scores_padded)],
                                    dim=-1)

        output = (output_padded, lengths)
        if output_hidden:
            output += (hidden,)
        return output


class EmbeddingPreprocessor(nn.Module):
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 512, multi_speaker: bool = False):
        super().__init__()
        self.size_embedding = embedding_dim
        self.size_hidden = hidden_dim
        self.silu = torch.nn.SiLU()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        if multi_speaker:
            self.max_pool = nn.MaxPool1d(kernel_size=hidden_dim)
            self.embedding_preprocessor = nn.Sequential(self.fc1, self.silu, self.max_pool, self.fc2)
        else:
            self.embedding_preprocessor = nn.Sequential(self.fc1, self.silu, self.fc2)

    def forward(self, embedding):
        return self.embedding_preprocessor(embedding)


class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim=64, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.film = FiLM(embedding_dim=embedding_dim, input_dim=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, embedding):
        residual = x
        x = self.fc1(x)
        x = self.silu(x)
        x = self.film(x, embedding)
        x = self.fc2(x)
        return x + residual


class PVAD1ET3(nn.Module):
    """Personal VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers=2, out_dim=3, embedding_dim=256, embedding_hidden_dim=512):
        """PersonalVAD class initializer.
        Args:
            input_dim (int): Input feature vector size.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers in the model.
            out_dim (int): Number of neurons in the output layer.
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.embedding_hidden_dim = embedding_hidden_dim

        # preprocess embeddings
        self.embedding_preprocessor = EmbeddingPreprocessor(embedding_dim=embedding_dim,
                                                            hidden_dim=embedding_hidden_dim)
        # modulation
        self.film = FiLMBlock(embedding_dim=embedding_dim, hidden_dim=input_dim)
        # LSTM encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Linear classification head
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, speaker_embeddings, lengths, hidden=None, output_hidden=None):
        """Personal VAD model forward pass method."""

        # Preprocess speaker_embeddings
        speaker_embeddings = self.embedding_preprocessor(speaker_embeddings)

        # Modulate input with speaker embeddings
        x_modulated = self.film(x, speaker_embeddings)

        # Pass features through lstm
        x_packed = pack_padded_sequence(x_modulated, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.encoder(x_packed, hidden)
        out_padded, lengths = pad_packed_sequence(out_packed, batch_first=True)

        # Project to output dimensionality
        out_padded = self.fc(out_padded)

        output = (out_padded, lengths)
        if output_hidden:
            output += (hidden,)
        return output


class PVAD1ET4(nn.Module):
    """Personal VAD model class. """

    def __init__(self, input_dim, hidden_dim, num_layers=2, out_dim=3, embedding_dim=256, common_layers=1):
        """PersonalVAD class initializer.
        Args:
            input_dim (int): Input feature vector size.
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers in the model.
            out_dim (int): Number of neurons in the output layer.
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        # define the model layers
        self.lstm_common = nn.LSTM(input_dim, hidden_dim, common_layers, batch_first=True)
        self.lstm_vad = nn.LSTM(hidden_dim, hidden_dim, num_layers - common_layers, batch_first=True)
        self.lstm_speaker_encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers - common_layers, batch_first=True)
        self.modulation = nn.Linear(embedding_dim, hidden_dim)
        self.fc_vad = nn.Linear(hidden_dim, 2)
        self.fc_speaker_encoder = nn.Linear(hidden_dim, 1)

        # similarity score scaling parameters
        self.similarity_weight = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.tensor(-5.0), requires_grad=True)

    def forward(self, x, speaker_embeddings, lengths, hidden=None, output_hidden=None):
        """Personal VAD model forward pass method."""
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # common feature extraction
        common_packed, common_hidden = self.lstm_common(x_packed, hidden)
        common_padded, _ = pad_packed_sequence(common_packed, batch_first=True)

        # VAD
        vad_packed, vad_hidden = self.lstm_vad(common_packed)
        vad_padded, _ = pad_packed_sequence(vad_packed, batch_first=True)
        vad_padded = self.fc_vad(vad_padded)

        # speaker verification
        speaker_modulated_common_padded = common_padded + self.modulation(speaker_embeddings).unsqueeze(1)
        speaker_modulated_common_packed = pack_padded_sequence(speaker_modulated_common_padded,
                                                               lengths.cpu(),
                                                               batch_first=True,
                                                               enforce_sorted=False)
        speaker_encoder_packed, speaker_encoder_hidden = self.lstm_speaker_encoder(speaker_modulated_common_packed)
        speaker_encoder_padded, _ = pad_packed_sequence(speaker_encoder_packed, batch_first=True)
        similarity_scores = self.fc_speaker_encoder(speaker_encoder_padded)
        similarity_scores = similarity_scores * self.similarity_weight + self.similarity_bias

        # Classification
        output = torch.stack([vad_padded[:, :, 0],
                              vad_padded[:, :, 1] * similarity_scores[:, :, 0],
                              vad_padded[:, :, 1] * (1 - similarity_scores[:, :, 0])],
                             dim=-1)

        output = (output, lengths)

        if output_hidden:
            hidden = [common_hidden, vad_hidden, speaker_encoder_hidden]
            output += (hidden,)
        return output


if __name__ == "__main__":
    from common.misc import count_parameters

    model = PVAD1ET22(input_dim=40, hidden_dim=64, embedding_dim=256, num_layers=2)
    parameters = count_parameters(model)

    features = torch.ones(32, 100, 40)
    lengths = torch.ones(32, ) * 100
    embedding = torch.randn(size=(32, 256))

    out = model(x=features, speaker_embeddings=embedding, lengths=lengths)

    print("done")
