from operator import itemgetter
from typing import Optional, Union

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from tqdm import tqdm

import wandb


def wandb_log_confusion_matrix(counts: np.ndarray, class_names, title: str = ""):
    n_classes = len(class_names)
    data = []
    for i in range(n_classes):
        for j in range(n_classes):
            data.append([class_names[i], class_names[j], counts[i, j]])

    fields = {
        "Actual": "Actual",
        "Predicted": "Predicted",
        "nPredictions": "nPredictions",
    }

    return wandb.plot_table(
        "wandb/confusion_matrix/v1",
        wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
        fields,
        {"title": title},
    )


def plot_confusion_matrix(confusion_matrix: np.ndarray):
    return ConfusionMatrixDisplay(confusion_matrix, display_labels=["ns", "tss", "ntss"])
