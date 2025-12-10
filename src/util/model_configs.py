from enum import Enum
from dataclasses import dataclass
from typing import Type, Dict
import torch

from src.data_loader.abstract_data_loader import AbstractDataLoader
from src.data_loader.autoencoder_data_loader import AutoEncoderDataLoader
from src.data_loader.stfpm_data_loader import STFPMDataLoader
from src.metrics.stfpm_metrics import STFPMValMetric, STFPMTrainLoss, \
    STFPMTestMetric
from src.model_evaluator.autoencoder_model_evaluator import \
    AutoencoderModelEvaluator
from src.model_evaluator.generic_model_evaluator import GenericModelEvaluator
from src.model_evaluator.stfpm_model_evaluator import STFPMModelEvaluator
from src.models.auto_encoder import AutoEncoderModel
from src.models.auto_encoder2dconv import AutoEncoderConv2DModel
from src.models.stfpm import STFPMModel
from src.util.constants import SADD_IMAGES_GOOD_CLEAN_PATH, \
    SADD_IMAGES_GOOD_STRIPES_PATH

"""
Some generic data classes to serve as configurations for the models to train. 
You can define your own following this scheme depending on your needs. You can 
use the x_parameter dictionaries in the GenericHyperparametersConfig to set 
arbitrary parameters for your models that will be passed through your model 
when using the generic model trainer class without changing it.
"""


@dataclass
class GenericClassesConfig:
    model: Type[torch.nn.Module]
    evaluator: Type[GenericModelEvaluator]
    data_loader: Type[AbstractDataLoader]
    train_loss: Type[torch.nn.Module]
    val_metric: Type[torch.nn.Module]
    test_metric: Type[torch.nn.Module]
    optimizer: Type[torch.optim.Optimizer]


@dataclass
class GenericHyperparametersConfig:
    batch_size: int
    num_epochs: int
    maximise_val_metric: bool
    maximise_test_metric: bool
    optimizer_parameters: Dict
    data_loader_parameters: Dict
    model_evaluator_parameters: Dict
    model_parameters: Dict


@dataclass
class GenericModelConfig:
    model_export_name: str
    classes: GenericClassesConfig
    hyperparameters: GenericHyperparametersConfig


class AUTOENCODERDataset(Enum):
    AUTOENCODER_SUBSET_A = [SADD_IMAGES_GOOD_CLEAN_PATH]
    AUTOENCODER_SUBSET_B = [SADD_IMAGES_GOOD_CLEAN_PATH,
                            SADD_IMAGES_GOOD_STRIPES_PATH]


class STFPMDataset(Enum):
    SHEAROGRAPHIE_SUBSET_A = [[SADD_IMAGES_GOOD_CLEAN_PATH], (192, 105)]
    SHEAROGRAPHIE_SUBSET_B = [[SADD_IMAGES_GOOD_CLEAN_PATH,
                               SADD_IMAGES_GOOD_STRIPES_PATH], (192, 105)]


class ModelTypes(Enum):
    AUTOENCODER_SUBSET_A = GenericModelConfig(
        model_export_name='ae',
        classes=GenericClassesConfig(
            model=AutoEncoderModel,
            evaluator=AutoencoderModelEvaluator,
            data_loader=AutoEncoderDataLoader,
            train_loss=torch.nn.MSELoss,
            val_metric=torch.nn.MSELoss,
            test_metric=torch.nn.MSELoss,
            optimizer=torch.optim.Adam
        ),
        hyperparameters=GenericHyperparametersConfig(
            batch_size=16,
            num_epochs=50,
            maximise_val_metric=False,
            maximise_test_metric=False,
            optimizer_parameters={
                'lr': 1e-3
            },
            data_loader_parameters={
                'dataset_name': AUTOENCODERDataset.AUTOENCODER_SUBSET_A.value,
                'img_x': 96,
                'img_y': 50
            },
            model_evaluator_parameters={
                'threshold': 0.005
            },
            model_parameters={}
        )
    )
    AUTOENCODER_SUBSET_B = GenericModelConfig(
        model_export_name='ae',
        classes=GenericClassesConfig(
            model=AutoEncoderModel,
            evaluator=AutoencoderModelEvaluator,
            data_loader=AutoEncoderDataLoader,
            train_loss=torch.nn.MSELoss,
            val_metric=torch.nn.MSELoss,
            test_metric=torch.nn.MSELoss,
            optimizer=torch.optim.Adam
        ),
        hyperparameters=GenericHyperparametersConfig(
            batch_size=16,
            num_epochs=50,
            maximise_val_metric=False,
            maximise_test_metric=False,
            optimizer_parameters={
                'lr': 1e-3
            },
            data_loader_parameters={
                'dataset_name': AUTOENCODERDataset.AUTOENCODER_SUBSET_B.value,
                'img_x': 96,
                'img_y': 50
            },
            model_evaluator_parameters={
                'threshold': 0.005
            },
            model_parameters={}
        )
    )
    AUTOENCODER2DCONV_SUBSET_A = GenericModelConfig(
        model_export_name='convae',
        classes=GenericClassesConfig(
            model=AutoEncoderConv2DModel,
            evaluator=AutoencoderModelEvaluator,
            data_loader=AutoEncoderDataLoader,
            train_loss=torch.nn.MSELoss,
            val_metric=torch.nn.MSELoss,
            test_metric=torch.nn.MSELoss,
            optimizer=torch.optim.Adam
        ),
        hyperparameters=GenericHyperparametersConfig(
            batch_size=16,
            num_epochs=50,
            maximise_val_metric=False,
            maximise_test_metric=False,
            optimizer_parameters={
                'lr': 1e-3
            },
            data_loader_parameters={
                'dataset_name': AUTOENCODERDataset.AUTOENCODER_SUBSET_A.value,
                'img_x': 192,
                'img_y': 192
            },
            model_evaluator_parameters={
                'threshold': 0.0002
            },
            model_parameters={}

        )
    )
    AUTOENCODER2DCONV_SUBSET_B = GenericModelConfig(
        model_export_name='convae',
        classes=GenericClassesConfig(
            model=AutoEncoderConv2DModel,
            evaluator=AutoencoderModelEvaluator,
            data_loader=AutoEncoderDataLoader,
            train_loss=torch.nn.MSELoss,
            val_metric=torch.nn.MSELoss,
            test_metric=torch.nn.MSELoss,
            optimizer=torch.optim.Adam
        ),
        hyperparameters=GenericHyperparametersConfig(
            batch_size=16,
            num_epochs=50,
            maximise_val_metric=False,
            maximise_test_metric=False,
            optimizer_parameters={
                'lr': 1e-3
            },
            data_loader_parameters={
                'dataset_name': AUTOENCODERDataset.AUTOENCODER_SUBSET_B.value,
                'img_x': 192,
                'img_y': 192
            },
            model_evaluator_parameters={
                'threshold': 0.0002
            },
            model_parameters={}

        )
    )
    STFPM_SUBSET_A = GenericModelConfig(
        model_export_name='stfpm',
        classes=GenericClassesConfig(
            model=STFPMModel,
            evaluator=STFPMModelEvaluator,
            data_loader=STFPMDataLoader,
            train_loss=STFPMTrainLoss,
            val_metric=STFPMValMetric,
            test_metric=STFPMTestMetric,
            optimizer=torch.optim.SGD
        ),
        hyperparameters=GenericHyperparametersConfig(
            batch_size=32,
            num_epochs=50,
            maximise_val_metric=False,
            maximise_test_metric=False,
            optimizer_parameters={
                'lr': 0.4,
                'momentum': 0.9,
                'weight_decay': 1e-4
            },
            data_loader_parameters={
                'load_into_memory': False,
                'dataset_name': STFPMDataset.SHEAROGRAPHIE_SUBSET_A.value[0]
            },
            model_evaluator_parameters={
                'scale_factor': 10.,
                'img_size': STFPMDataset.SHEAROGRAPHIE_SUBSET_A.value[1],
                'export_video': True
            },
            model_parameters={}
        )
    )
    STFPM_SUBSET_B = GenericModelConfig(
        model_export_name='stfpm',
        classes=GenericClassesConfig(
            model=STFPMModel,
            evaluator=STFPMModelEvaluator,
            data_loader=STFPMDataLoader,
            train_loss=STFPMTrainLoss,
            val_metric=STFPMValMetric,
            test_metric=STFPMTestMetric,
            optimizer=torch.optim.SGD
        ),
        hyperparameters=GenericHyperparametersConfig(
            batch_size=32,
            num_epochs=50,
            maximise_val_metric=False,
            maximise_test_metric=False,
            optimizer_parameters={
                'lr': 0.4,
                'momentum': 0.9,
                'weight_decay': 1e-4
            },
            data_loader_parameters={
                'load_into_memory': False,
                'dataset_name': STFPMDataset.SHEAROGRAPHIE_SUBSET_B.value[0]
            },
            model_evaluator_parameters={
                'scale_factor': 10.,
                'img_size': STFPMDataset.SHEAROGRAPHIE_SUBSET_B.value[1],
                'export_video': True
            },
            model_parameters={}
        )
    )
