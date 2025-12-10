import os
import pathlib

import torch
import numpy as np

from pathlib import Path
from glob import glob
from typing import Tuple, List
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data_loader.abstract_data_loader import AbstractDataLoader
from src.util.constants import SAD_DATASET_PATH, SADD_IMAGES_FAULTY_PATH
from src.util.load_labels import load_original_labels


class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, images: List,
                 transform: torch.nn.Module | transforms.Compose = None):
        super().__init__()
        self._paths = images
        self._len = len(self._paths)
        self._transform = transform

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self._paths[index]
        image = Image.open(path)
        image = self._transform(image)

        # Input, Label == Image
        return image, image


class AutoEncoderTestDataset(AutoEncoderDataset):

    def __init__(self, images: List,
                 transform: torch.nn.Module | transforms.Compose = None):
        super().__init__(images=images, transform=transform)

        self._labels = load_original_labels()

    def __getitem__(self, index: int) \
            -> Tuple[torch.Tensor, torch.Tensor, List, np.array]:
        path = self._paths[index]
        image_raw = Image.open(path)
        image = self._transform(image_raw)
        width, height = image_raw.size
        image_raw = image_raw.resize((int(width / 10), int(height / 10)))
        image_labels = self._labels[Path(path).name] if Path(
            path).name in self._labels.keys() else 'no fault'

        # Input, Label == Image
        return image, image, image_labels, np.array(image_raw)


class AutoEncoderDataLoader(AbstractDataLoader):

    def __init__(self, batch_size: int, img_x: int, img_y: int,
                 dataset_name: str):
        super().__init__(batch_size=batch_size)
        self._dataset_name = dataset_name
        self._transform = transforms.Compose([
            transforms.Resize([img_y, img_x]),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])

    def get_data_loaders(self) -> Tuple[
        DataLoader, DataLoader, List[DataLoader]]:

        image_list_train = []
        image_list_val = []
        image_list_test_good = []

        for subset in self._dataset_name:
            image_list_train += sorted(
                glob(str(subset.joinpath('train', '*.png'))))
            image_list_val += sorted(
                glob(str(subset.joinpath('val', '*.png'))))
            image_list_test_good += sorted(
                glob(str(subset.joinpath('test', '*.png'))))


        image_list_test_bad = sorted(glob(os.path.join(SADD_IMAGES_FAULTY_PATH,
                                                       'test', '*.png')))

        train_dataset = AutoEncoderDataset(image_list_train,
                                           transform=self._transform)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self._batch_size, shuffle=True,
                                  drop_last=False)

        val_dataset = AutoEncoderDataset(image_list_val,
                                         transform=self._transform)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size,
                                shuffle=False, drop_last=False)

        test_good_dataset = AutoEncoderTestDataset(image_list_test_good,
                                                   transform=self._transform)
        test_good_loader = DataLoader(test_good_dataset, batch_size=1,
                                      shuffle=False, drop_last=False)
        test_bad_dataset = AutoEncoderTestDataset(image_list_test_bad,
                                                  transform=self._transform)
        test_bad_loader = DataLoader(test_bad_dataset, batch_size=1,
                                     shuffle=False, drop_last=False)

        return train_loader, val_loader, [test_good_loader, test_bad_loader]
