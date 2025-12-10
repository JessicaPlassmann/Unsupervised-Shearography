import os
import torch
import random
import numpy as np


def set_deterministic_behavior(seed: int) -> None:
    """
    Call this function to seed various RNGs for torch, numpy etc. This should
    enable the training of the models to be deterministic. You might want to
    do a few short test runs to check if it works on your setup.

    :param seed: An int to use as a seed for the random number generator.
    """

    random.seed(a=seed)
    np.random.seed(seed=seed)

    torch.manual_seed(seed=seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # This line can help mitigating some error I got while trying to train
    #   an individual model while setting deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
