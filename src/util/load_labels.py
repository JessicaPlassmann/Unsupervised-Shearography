import glob
import json
import pathlib
import pandas as pd

from typing import Dict

from src.util.constants import EXPERT_LABELS_CSV_PATH


def load_original_labels() -> Dict:
    """
    Load the ground truth data for the autoencoder.

    :return: A dictionary containing the dataset image names as a key and the
        labels as a value.
    """

    labels_dict = {}

    label_df = pd.read_csv(EXPERT_LABELS_CSV_PATH, delimiter=';')

    for file_name in list(label_df['image'].unique()):
        labels_dict[file_name] = list(
            label_df[label_df['image'] == file_name]['class'])

    return labels_dict
