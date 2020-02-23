import numpy as np
import torch
from torch import Tensor


def generate_measurement_operators(mode, m, n):
    if mode == "gaussian":
        """values that parameterize the measurement model. 
        This could be the measurement matrix itself or the 
        random mask with coded diffraction patterns."""
        A_val = np.float32(1.0 / np.sqrt(m) * np.random.randn(m, n))

        def A_handle(A_vals, x):
            return torch.mm(A_vals, x)

        def At_handle(A_vals: Tensor, z):
            return torch.mm(torch.t(A_vals), z)

        return [A_handle, At_handle, A_val]

    else:
        raise ValueError("Measurement mode not recognized")


def train_ldamp(
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
    **kwargs
):
    pass
