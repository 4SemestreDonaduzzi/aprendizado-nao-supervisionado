import random

import numpy as np
import torch

from autoencoder import set_seed


def test_set_seed_repeats_python_numpy_and_torch_sequences():
    set_seed(17)
    first = (random.random(), np.random.rand(), torch.rand(3))

    set_seed(17)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
