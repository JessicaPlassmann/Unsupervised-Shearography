from abc import ABC, abstractmethod
from typing import Tuple, Sequence
from torch.utils.data import DataLoader


class AbstractDataLoader(ABC):
    """
    If you implement your own data loader for a new custom model, you should
    inherit from this one.
    """

    @abstractmethod
    def __init__(self, batch_size: int, **kwargs):
        self._batch_size = batch_size

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader,
    DataLoader | None |
    Sequence[DataLoader]]:
        pass
