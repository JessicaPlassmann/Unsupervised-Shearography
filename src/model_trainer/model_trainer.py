import dataclasses
import json
import logging
import math
import os
import torch

from datetime import datetime
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.util.constants import RESULTS_PATH
from src.util.model_configs import GenericModelConfig

# Set general logging config
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d [%(levelname)s] [%('
                           'threadName)s] '
                           '(%(filename)s:%(lineno)d) -- %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class ModelTrainer:
    """
    Generic trainer class to train a model. You can use the configuration
    classes in util to define and implement your own custom model and
    pass it to this class. The model should be a torch nn and it will work.
    """

    def __init__(self, config: GenericModelConfig, export_folder: str = None):
        # Hyperparameters
        self._config = config
        self._export_folder = export_folder
        self._batch_size = self._config.hyperparameters.batch_size
        self._num_epochs = self._config.hyperparameters.num_epochs

        # Logger
        self._logger = logging.getLogger(__name__)

        # Data
        self._data_loader = self._config.classes.data_loader(
            batch_size=self._batch_size,
            **self._config.hyperparameters.data_loader_parameters)
        self._train_loader, self._val_loader, self._test_loader = (
            self._data_loader.get_data_loaders())

        self._latest_export_path = None

        # Model and Loss
        self._model = self._config.classes.model(
            **self._config.hyperparameters.model_parameters)

        if torch.cuda.is_available():
            self._logger.info('CUDA available, load model to GPU.')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model.to(device)

        self._model_evaluator = self._config.classes.evaluator(
            **self._config.hyperparameters.model_evaluator_parameters) \
            if callable(self._config.classes.evaluator) else None
        self._train_loss = self._config.classes.train_loss()
        self._val_metric = self._config.classes.val_metric()
        self._test_metric = self._config.classes.test_metric()
        self._optimizer = self._config.classes.optimizer(
            self._model.parameters(),
            **self._config.hyperparameters.optimizer_parameters)

    def train(self, seed: int = None):
        formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        export_folder = formatted_timestamp if (
                self._export_folder is None) else self._export_folder
        export_folder = RESULTS_PATH.joinpath(
            self._config.model_export_name, export_folder)
        self._latest_export_path = export_folder
        os.makedirs(export_folder, exist_ok=True)

        # Context manager to combine logging and tqdm progress bar
        with logging_redirect_tqdm():
            training_info = []

            batch_train_count = len(self._train_loader)
            batch_val_count = len(self._val_loader)

            self._logger.info(
                f'Start training for model '
                f'{self._model.__class__.__name__}, timestamp '
                f'{formatted_timestamp}.')
            self._logger.info(
                f'Train for {self._num_epochs} epochs, using batch size '
                f'{self._batch_size}.')

            best_loss = -math.inf if \
                self._config.hyperparameters.maximise_val_metric else math.inf

            # Walrus operator (:=) to update tqdm progress bar (postfix)
            for epoch in (pbar_epoch := tqdm(range(self._num_epochs),
                                             bar_format='{l_bar}{bar:10}{'
                                                        'r_bar}{bar:-10b}',
                                             position=0, leave=True,
                                             desc='Progress Total')):
                running_loss_train = 0
                running_metric_val = 0

                self._model.train()

                for batch in (pbar_batch_train := tqdm(self._train_loader,
                                                       bar_format='{l_bar}{'
                                                                  'bar:10}{'
                                                                  'r_bar}{'
                                                                  'bar:-10b}',
                                                       position=1, leave=False,
                                                       desc='Progress Train')):
                    data, labels = batch

                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    data = data.to(device)
                    labels = labels.to(device)

                    predictions = self._model(data)

                    loss_train = self._train_loss(predictions, labels)
                    pbar_batch_train.set_postfix({
                        'batch_train_loss': loss_train.item()})
                    running_loss_train += loss_train.item()

                    self._optimizer.zero_grad()
                    loss_train.backward()
                    self._optimizer.step()

                self._model.eval()

                for batch in (pbar_batch_val := tqdm(self._val_loader,
                                                     bar_format='{l_bar}{'
                                                                'bar:10}{'
                                                                'r_bar}{'
                                                                'bar:-10b}',
                                                     position=1,
                                                     leave=False,
                                                     desc='Progress Val  ')):
                    data, labels = batch

                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    data = data.to(device)
                    labels = labels.to(device)

                    predictions = self._model(data)

                    metric_val = self._val_metric(predictions, labels)
                    pbar_batch_val.set_postfix({
                        'batch_val_metric': metric_val.item()})
                    running_metric_val += metric_val.item()

                running_loss_train /= batch_train_count
                running_metric_val /= batch_val_count
                pbar_epoch.set_postfix({
                    'epoch_train_loss': running_loss_train,
                    'epoch_val_metric': running_metric_val})
                training_info.append({
                    'epoch': epoch + 1,
                    'train_loss': running_loss_train,
                    'val_metric': running_metric_val
                })

                if self._config.hyperparameters.maximise_val_metric:
                    if running_metric_val > best_loss:
                        best_loss = running_metric_val
                        torch.save(self._model.state_dict(),
                                   export_folder.joinpath('best_model.pth'))
                else:
                    if running_metric_val < best_loss:
                        best_loss = running_metric_val
                        torch.save(self._model.state_dict(),
                                   export_folder.joinpath('best_model.pth'))

            self._logger.info('Training finished.')

            # Dump logfile dictionary to json
            log_file = {
                'timestamp': formatted_timestamp,
                'seed': seed,
                'export_folder': export_folder.stem,
                'config': self.get_export_config(),
                'training_info': training_info
            }

            # TODO fix this; use string instead, but how to handle in other
            #   parts of the code?
            # Fix: just use a string in config as key to map
            # TODO: Maybe just remove the parameter from the hyperparameters
            #   and just make it a fixed property of the specific Dataset Class

            log_file['config']['hyperparameters']['data_loader_parameters'][
                'dataset_name'] = str(log_file['config']['hyperparameters'][
                                          'data_loader_parameters'][
                                          'dataset_name'])

            with open(export_folder.joinpath('training_info.json'), 'w') as f:
                json.dump(log_file, f, indent=2)

            self._logger.info(f'Exported logs to {export_folder}')

    def get_export_config(self):
        export_config = dataclasses.asdict(self._config)
        export_config['classes'] = {k: v.__name__ for k, v in
                                    export_config['classes'].items()}
        return export_config

    def evaluate(self, checkpoint: str = None, export_folder: str = None):
        with logging_redirect_tqdm():
            checkpoint = str(
                self._latest_export_path.joinpath('best_model.pth')) \
                if checkpoint is None else checkpoint
            self._model_evaluator.eval(model=self._model,
                                       test_loader=self._test_loader,
                                       eval_metric=self._test_metric,
                                       logger=self._logger,
                                       checkpoint=checkpoint,
                                       export_name=self._config.model_export_name,
                                       export_folder=export_folder)

    def get_eval_metrics(self, checkpoint: str = None):
        with logging_redirect_tqdm():
            checkpoint = str(
                self._latest_export_path.joinpath('best_model.pth')) \
                if checkpoint is None else checkpoint
            return self._model_evaluator.compute_metrics(model=self._model,
                                                         test_loader=self._test_loader,
                                                         eval_metric=self._test_metric,
                                                         logger=self._logger,
                                                         checkpoint=checkpoint)
