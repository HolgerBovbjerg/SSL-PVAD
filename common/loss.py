import torch
from torch import nn
from torch.nn.functional import one_hot
from typing import Optional


class WeightedPairwiseLoss(nn.Module):
    """Weighted pairwise loss implementation for three classes.
    The weight pairs are interpreted as follows:
    [<ns, ntss> ; <tss, ns> ; <ntss, tss>]
    Target labels contain indices, the model output is a tensor of probabilities for each class.
    (ns, tss, ntss) -> {0, 1, 2}

    Attributes:
        weights: tensor containing weight pairs [<ns, ntss> ; <tss, ns> ; <ntss, tss>]
    """

    def __init__(self, ignore_index: int = -1, weights=None):
        """Initializer for LabelSmoothingLoss.
        Args:
            weights: tuple containing pairwise weights
        """
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        self.weights = torch.tensor(weights, device=self.device)
        assert len(weights) == 3, "Weighted Pairwise Loss only supports three classes."
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward function.
        Args:
            pred (torch.Tensor): Model predictions, of shape (batch_size, num_classes).
            target (torch.Tensor): Target tensor of shape (batch_size).
        Returns:
            torch.Tensor: Loss.
        """
        output = torch.exp(pred).transpose(-1, -2)
        target_non_negative = torch.clone(target)
        target_non_negative[target_non_negative < 0] = 0

        ignore_mask = (target != self.ignore_index).unsqueeze(-1)
        label_mask = torch.logical_and(one_hot(target_non_negative, num_classes=3), ignore_mask)  # boolean mask

        label_mask_r1 = torch.roll(label_mask, shifts=1, dims=-1)  # e.g., if tss, then ntss
        label_mask_r2 = torch.roll(label_mask, shifts=2, dims=-1)  # e.g., if tss, then ns

        # Get the probability of the actual label and the other two classes
        actual = torch.masked_select(output, mask=label_mask)
        plus_one = torch.masked_select(output, mask=label_mask_r1)
        minus_one = torch.masked_select(output, mask=label_mask_r2)

        # Arrays of the first pair weight and the second pair weight used in the equation
        w1 = torch.masked_select(self.weights, mask=label_mask)  # e.g., if tss, w1 is <tss, ns>
        w2 = torch.masked_select(self.weights, mask=label_mask_r1)  # e.g., if tss, w2 is <ntss, tss>

        # Compute partial loss for the two pair combination. # e.g., if tss, then <tss, ns> and <ntss, tss>
        first_pair = w1 * torch.log(actual / (actual + minus_one))
        second_pair = w2 * torch.log(actual / (actual + plus_one))

        # Negative mean value for the two pairs
        weighted_pairwise_loss = -0.5 * (first_pair + second_pair)

        # sum and average for minibatch
        return torch.mean(weighted_pairwise_loss)


def get_loss(name: str, weights: Optional[list[float, float, float]] = None, ignore_index=-1):
    if name == "weighted_pairwise":
        return WeightedPairwiseLoss(weights=weights, ignore_index=ignore_index)
    elif name == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        raise ValueError(f"Loss with name: {name}, not supported.")
