import logging
import os
import torch

from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.data_loader.stfpm_data_loader import STFPMDataLoader
from src.model_trainer.model_trainer import ModelTrainer
from src.util.constants import RESULTS_PATH, SADD_IMAGES_GOOD_CLEAN_PATH, \
    SADD_IMAGES_GOOD_STRIPES_PATH


class ROCModelEvaluator:
    def __init__(self, model_yolo_expert, data_subset):

        self._model_yolo_expert = model_yolo_expert
        self._data_subset = data_subset
        self._logger = logging.getLogger(__name__)

    def eval_yolo(self, model):
        if self._data_subset == 'a':
            datasets = [SADD_IMAGES_GOOD_CLEAN_PATH]
        elif self._data_subset == 'b':
            datasets = [SADD_IMAGES_GOOD_CLEAN_PATH,
                        SADD_IMAGES_GOOD_STRIPES_PATH]
        else:
            raise NotImplementedError('Data subset must be either "a" or "b"')

        data_loader = STFPMDataLoader(batch_size=1, load_into_memory=False,
                                      dataset_name=datasets)
        results_good = []
        results_faulty = []

        _, _, test_loader = data_loader.get_data_loaders()
        test_loader_good = test_loader[0]
        test_loader_faulty = test_loader[1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        for batch in tqdm(test_loader_good,
                          bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                          position=1, leave=True,
                          desc='Progress Good Imgs  '):
            _, _, img_path = batch
            result = model(img_path)
            results_good.append(result)

        for batch in tqdm(test_loader_faulty,
                          bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                          position=1, leave=True,
                          desc='Progress Faulty Imgs'):
            _, _, img_path = batch
            result = model(img_path)
            results_faulty.append(result)

        conf_good = []
        conf_faulty = []

        for entry in results_good:
            confs = entry[0].boxes.conf.tolist()

            if len(confs) == 0:
                confs = [0]

            conf_good.append(max(confs))

        for entry in results_faulty:
            confs = entry[0].boxes.conf.tolist()

            if len(confs) == 0:
                confs = [0]

            conf_faulty.append(max(confs))

        return conf_good, conf_faulty


class ROCModelEvaluatorWeaklySupervised(ROCModelEvaluator):
    def __init__(self, model_yolo_expert, data_subset,
                 model_yolo_cells_dino, model_yolo_cells_sam,
                 model_yolo_twocircles_dino, model_yolo_twocircles_sam):
        super().__init__(model_yolo_expert, data_subset)
        self._model_yolo_cells_dino = model_yolo_cells_dino
        self._model_yolo_cells_sam = model_yolo_cells_sam
        self._model_yolo_twocircles_dino = model_yolo_twocircles_dino
        self._model_yolo_twocircles_sam = model_yolo_twocircles_sam

    def eval(self, export_folder: str = None):
        with (((logging_redirect_tqdm()))):
            show_imgs = False

            formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            export_folder = formatted_timestamp \
                if export_folder is None else export_folder
            export_path = RESULTS_PATH.joinpath(
                'multi_model_binary_eval_weakly_supervised', export_folder)
            self._logger.info(f'Evaluate various models with timestamp '
                              f'{formatted_timestamp}')

            os.makedirs(export_path, exist_ok=True)

            self._logger.info('Evaluate YOLO BBox (Expert Annotations)')
            results_yolo_expert_good, results_yolo_expert_faulty = (
                self.eval_yolo(model=self._model_yolo_expert))

            self._logger.info('Evaluate YOLO BBox (Cells, DINO)')
            results_yolo_cells_dino_good, results_yolo_cells_dino_faulty = (
                self.eval_yolo(model=self._model_yolo_cells_dino))

            self._logger.info('Evaluate YOLO BBox (Cells, SAM)')
            results_yolo_cells_sam_good, results_yolo_cells_sam_faulty = (
                self.eval_yolo(model=self._model_yolo_cells_sam))

            self._logger.info('Evaluate YOLO BBox (Two Circles, DINO)')
            (results_yolo_twocircles_dino_good,
             results_yolo_twocircles_dino_faulty) = self.eval_yolo(
                model=self._model_yolo_twocircles_dino)

            self._logger.info('Evaluate YOLO BBox (Two Circles, SAM)')
            (results_yolo_twocircles_sam_good,
             results_yolo_twocircles_sam_faulty) = self.eval_yolo(
                model=self._model_yolo_twocircles_sam)

            if self._data_subset == 'a':
                datasets = [SADD_IMAGES_GOOD_CLEAN_PATH]
            elif self._data_subset == 'b':
                datasets = [SADD_IMAGES_GOOD_CLEAN_PATH,
                            SADD_IMAGES_GOOD_STRIPES_PATH]
            else:
                raise NotImplementedError(
                    'Data subset must be either "a" or "b"')

            data_loader = STFPMDataLoader(batch_size=1, load_into_memory=False,
                                          dataset_name=datasets)

            _, _, test_loader = data_loader.get_data_loaders()
            test_loader_good = test_loader[0]
            test_loader_faulty = test_loader[1]

            y_test = [0] * len(test_loader_good) + [1] * len(
                test_loader_faulty)
            y_score_yolo_conf_cells_dino = (
                    results_yolo_cells_dino_good +
                    results_yolo_cells_dino_faulty)
            y_score_yolo_conf_cells_sam = (
                    results_yolo_cells_sam_good +
                    results_yolo_cells_sam_faulty)
            y_score_yolo_conf_twocircles_dino = (
                    results_yolo_twocircles_dino_good +
                    results_yolo_twocircles_dino_faulty)
            y_score_yolo_conf_twocircles_sam = (
                    results_yolo_twocircles_sam_good +
                    results_yolo_twocircles_sam_faulty)
            y_score_yolo_conf_hand = results_yolo_expert_good + \
                                     results_yolo_expert_faulty

            ### PLOT ALL YOLO MODELS ###
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_predictions(
                y_test,
                y_score_yolo_conf_hand,
                name='YOLOv8 (Expert Annotations)',
                plot_chance_level=False,
                ax=ax)
            PrecisionRecallDisplay.from_predictions(
                y_test,
                y_score_yolo_conf_twocircles_dino,
                name='YOLOv8 (Two Circles, DINO)',
                plot_chance_level=False,
                ax=ax)
            PrecisionRecallDisplay.from_predictions(
                y_test,
                y_score_yolo_conf_twocircles_sam,
                name='YOLOv8 (Two Circles, SAM)',
                plot_chance_level=False,
                ax=ax)
            PrecisionRecallDisplay.from_predictions(
                y_test,
                y_score_yolo_conf_cells_sam,
                name='YOLOv8 (Cells, SAM)',
                plot_chance_level=False,
                ax=ax)
            PrecisionRecallDisplay.from_predictions(
                y_test,
                y_score_yolo_conf_cells_dino,
                name='YOLOv8 (Cells, DINO)',
                plot_chance_level=True,
                ax=ax)
            ax.set_title('Binary (Good/Faulty) PR-Curve')
            fig.savefig(export_path.joinpath('yolo_models_pr_curve.svg'))
            if show_imgs:
                plt.show()

            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_score_yolo_conf_hand,
                                             name='YOLOv8 (Expert '
                                                  'Annotations)',
                                             ax=ax)
            RocCurveDisplay.from_predictions(y_test,
                                             y_score_yolo_conf_twocircles_dino,
                                             name='YOLOv8 (Two Circles, DINO)',
                                             ax=ax)
            RocCurveDisplay.from_predictions(y_test,
                                             y_score_yolo_conf_twocircles_sam,
                                             name='YOLOv8 (Two Circles, SAM)',
                                             ax=ax)
            RocCurveDisplay.from_predictions(y_test,
                                             y_score_yolo_conf_cells_sam,
                                             name='YOLOv8 (Cells, SAM)',
                                             ax=ax)
            RocCurveDisplay.from_predictions(y_test,
                                             y_score_yolo_conf_cells_dino,
                                             name='YOLOv8 (Cells, DINO)',
                                             ax=ax)
            ax.set_title('Binary (Good/Faulty) ROC-Curve')
            fig.savefig(export_path.joinpath('yolo_models_tpfp_curve.svg'))
            if show_imgs:
                plt.show()


class ROCModelEvaluatorUnsupervised(ROCModelEvaluator):
    def __init__(self, model_ae: ModelTrainer, model_convae: ModelTrainer,
                 model_stfpm: ModelTrainer, model_yolo_expert, data_subset):
        super().__init__(model_yolo_expert, data_subset)
        self._model_ae = model_ae
        self._model_convae = model_convae
        self._model_stfpm = model_stfpm

    def eval(self, checkpoint_ae, checkpoint_convae, checkpoint_stfpm,
             export_folder: str = None):
        with (((logging_redirect_tqdm()))):
            show_imgs = False

            formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            export_folder = formatted_timestamp \
                if export_folder is None else export_folder
            export_path = RESULTS_PATH.joinpath(
                'multi_model_binary_eval_unsupervised', export_folder)
            self._logger.info(f'Evaluate various models with timestamp '
                              f'{formatted_timestamp}')

            os.makedirs(export_path, exist_ok=True)

            self._logger.info('Evaluate AE')
            _, _, _, _, _, _, _, _, loss_ae_faulty, loss_ae_good = (
                self._model_ae.get_eval_metrics(checkpoint_ae))

            self._logger.info('Evaluate ConvAE')
            _, _, _, _, _, _, _, _, loss_convae_faulty, loss_convae_good = (
                self._model_convae.get_eval_metrics(checkpoint_convae))

            self._logger.info('Evaluate STFPM')
            means_faulty, means_good, peaks_faulty, peaks_good, _, _ = (
                self._model_stfpm.get_eval_metrics(checkpoint_stfpm))

            self._logger.info('Evaluate YOLO BBox (Expert Annotations)')
            results_yolo_expert_good, results_yolo_expert_faulty = \
                self.eval_yolo(model=self._model_yolo_expert)

            y_test = [0] * len(peaks_good) + [1] * len(peaks_faulty)
            y_score_stfpm_peaks = peaks_good + peaks_faulty
            y_score_stfpm_means = means_good + means_faulty
            y_score_ae_loss = loss_ae_good + loss_ae_faulty
            y_score_convae_loss = loss_convae_good + loss_convae_faulty
            y_score_yolo_conf_hand = results_yolo_expert_good + \
                                     results_yolo_expert_faulty

            ### PLOT COMPARE DIFFERENT MODELS ###
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_predictions(
                y_test,
                y_score_yolo_conf_hand,
                name='YOLOv8 (Expert Annotations)',
                plot_chance_level=False,
                ax=ax)
            PrecisionRecallDisplay.from_predictions(
                y_test,
                y_score_stfpm_peaks,
                name='STFPM (Peaks)',
                plot_chance_level=False,
                ax=ax)
            PrecisionRecallDisplay.from_predictions(
                y_test,
                y_score_stfpm_means,
                name='STFPM (Means)',
                plot_chance_level=False,
                ax=ax)
            PrecisionRecallDisplay.from_predictions(
                y_test,
                y_score_convae_loss,
                name='ConvAE',
                plot_chance_level=False,
                ax=ax)
            PrecisionRecallDisplay.from_predictions(
                y_test,
                y_score_ae_loss,
                name='AE',
                plot_chance_level=True,
                ax=ax)
            ax.set_title('Binary (Good/Faulty) PR-Curve')
            fig.savefig(export_path.joinpath('dif_models_pr_curve.svg'))
            if show_imgs:
                plt.show()

            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, y_score_yolo_conf_hand,
                                             name='YOLOv8 (Expert '
                                                  'Annotations)',
                                             ax=ax)
            RocCurveDisplay.from_predictions(y_test, y_score_stfpm_peaks,
                                             name='STFPM (Peaks)', ax=ax)
            RocCurveDisplay.from_predictions(y_test, y_score_stfpm_means,
                                             name='STFPM (Means)', ax=ax)
            RocCurveDisplay.from_predictions(y_test, y_score_convae_loss,
                                             name='ConvAE', ax=ax)
            RocCurveDisplay.from_predictions(y_test, y_score_ae_loss,
                                             name='AE', ax=ax)
            ax.set_title('Binary (Good/Faulty) ROC-Curve')
            fig.savefig(export_path.joinpath('dif_models_tpfp_curve.svg'))
            if show_imgs:
                plt.show()
