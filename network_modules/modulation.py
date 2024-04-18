import torch
from torch import nn


class FiLMGenerator(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, 2 * input_dim)

    def forward(self, embedding):
        film_parameters = self.linear(embedding).view(embedding.size(0), 2, -1)
        beta = film_parameters[:, 0]
        gamma = film_parameters[:, 1]
        return beta, gamma


class FiLM(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(FiLM, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.FiLM_generator = FiLMGenerator(embedding_dim=embedding_dim, input_dim=input_dim)

    def forward(self, x, embedding):
        beta, gamma = self.FiLM_generator(embedding=embedding)
        beta = beta.unsqueeze(1)
        gamma = gamma.unsqueeze(1)
        return gamma * x + beta


class BiasModulator(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(BiasModulator, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.linear = nn.Linear(embedding_dim, input_dim)

    def forward(self, x, embedding):
        return x + self.linear(embedding).unsqueeze(1)


class ScaleModulator(nn.Module):
    def __init__(self, embedding_dim, input_dim):
        super(ScaleModulator, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.linear = nn.Linear(embedding_dim, input_dim)

    def forward(self, x, embedding):
        return x * self.linear(embedding).unsqueeze(1)
