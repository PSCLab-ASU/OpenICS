import numpy as np
import torch
from torch import Tensor


def ldamp(
    sensing="gaussian",
    reconstruction="ldamp",
    stage="training",
    default=True,
    dataset="",
    input_channel=1,
    input_width=128,
    input_height=128,
    m=10,
    n=100,
    specifics=None,
):
    if specifics is None:
        specifics = {"max_n_DAMP_layers": 10, "start_layer": 1}
    if stage == "training":
        for n_DAMP_layers in range(
            specifics["start_layer"], specifics["max_n_DAMP_layers"] + 1
        ):
            n_layers_trained = n_DAMP_layers
            theta = []

        # A = np.float32(1.0 / np.sqrt(m) * np.random.randn(m, n))
        # y = torch.mm(A, )
