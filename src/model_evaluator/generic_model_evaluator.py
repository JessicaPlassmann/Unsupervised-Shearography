import logging
from typing import List
import torch
from torch.utils.data import DataLoader


class GenericModelEvaluator:
    """
    Generic model evaluator. If you implement your own models, you should
    inherit from this class.
    """

    def __init__(self, **kwargs):
        pass

    def eval(self, model: torch.nn.Module,
             test_loader: DataLoader | List[DataLoader],
             eval_metric: torch.nn.Module, logger: logging.Logger,
             checkpoint: str, export_name: str, export_folder: str = None):
        pass

    def compute_metrics(self, model: torch.nn.Module,
                        test_loader: DataLoader | List[DataLoader],
                        eval_metric: torch.nn.Module, logger: logging.Logger,
                        checkpoint: str):
        pass
