import glob
import itertools
import json
import logging
import os
import pathlib
import torch
import cv2
import numpy as np
import pandas as pd

from typing import List
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from datetime import datetime
from torch import tensor
from torchvision.ops import box_iou
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.data_loader.stfpm_data_loader import STFPMDataLoader
from src.util.constants import RESULTS_PATH, SADD_IMAGES_FAULTY_PATH, \
    EXPERT_LABELS_CSV_PATH, SADD_IMAGES_GOOD_CLEAN_PATH, \
    SADD_IMAGES_GOOD_STRIPES_PATH


class IoUModelEvaluator:
    def __init__(self, model_yolo_expert, data_subset):
        self._model_yolo_expert = model_yolo_expert
        self._data_subset = data_subset
        self._dataset_paths = [
            SADD_IMAGES_GOOD_CLEAN_PATH] if self._data_subset == 'a' else [
            SADD_IMAGES_GOOD_CLEAN_PATH, SADD_IMAGES_GOOD_STRIPES_PATH]
        self._logger = logging.getLogger(__name__)

    def get_labels(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        imgs_test = glob.glob(
            str(SADD_IMAGES_FAULTY_PATH.joinpath('test', '*.png')))

        labels_to_ignore = ['Wave', 'End', 'Edge', 'Bubble']
        imgs_labels = []

        label_df = pd.read_csv(EXPERT_LABELS_CSV_PATH, delimiter=';')

        for img_list in imgs_test:
            file_name = pathlib.Path(img_list).name

            img_labels = []

            for label in label_df[label_df['image'] == file_name].to_dict(
                    orient='records'):
                if label['class'] not in labels_to_ignore:
                    points = [label['xmin'], label['ymin'], label['xmax'],
                              label['ymax']]
                    img_labels.append(points)

            imgs_labels.append(tensor(img_labels, device=device))

        return imgs_labels

    def eval_yolo(self, model):
        data_loader = STFPMDataLoader(batch_size=1, load_into_memory=False,
                                      dataset_name=self._dataset_paths)
        results = []

        _, _, test_loader = data_loader.get_data_loaders()
        test_loader_faulty = test_loader[1]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        for batch in tqdm(test_loader_faulty,
                          bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                          position=1, leave=True,
                          desc='Progress Faulty Imgs'):
            _, _, img_path = batch
            result = model(img_path)
            results.append(result)

        bboxes = []
        probs = []
        classes = []

        for entry in results:
            bboxes.append(entry[0].boxes.xyxy)
            probs.append(entry[0].boxes.conf)
            classes.append(entry[0].boxes.cls)

        return [bboxes, probs, classes]

    def parse_labels(self, labels, device):
        parsed_labels = []

        for i, _ in enumerate(labels):
            parsed_label = {
                'boxes': labels[i],
                'labels': torch.tensor([0] * labels[i].size()[0],
                                       device=device)
            }
            parsed_labels.append(parsed_label)

        return parsed_labels

    def parse_yolo_results(self, results_to_parse):
        parsed_results = []

        for i in range(len(results_to_parse[0])):
            parsed_result = {
                'boxes': results_to_parse[0][i],
                'scores': results_to_parse[1][i],
                'labels': results_to_parse[2][i].int()
            }
            parsed_results.append(parsed_result)

        return parsed_results

    def get_ious(self, bboxes, device, y):
        ious = []

        for entry in zip(y, bboxes):
            if entry[0].size()[0] == 0:
                ious.append(tensor(1.0, device=device))
            elif entry[1].size()[0] == 0:
                ious.append(tensor(0.0, device=device))
            else:
                iou_matrix = box_iou(entry[0], entry[1])
                ious.append(torch.mean(torch.amax(iou_matrix, dim=1)))

        return ious


class IoUModelEvaluatorWeaklySupervised(IoUModelEvaluator):
    def __init__(self, model_yolo_expert, data_subset, model_yolo_cells_dino,
                 model_yolo_cells_sam, model_yolo_twocircles_dino,
                 model_yolo_twocircles_sam):
        super().__init__(model_yolo_expert, data_subset)
        self._model_yolo_cells_dino = model_yolo_cells_dino
        self._model_yolo_cells_sam = model_yolo_cells_sam
        self._model_yolo_twocircles_dino = model_yolo_twocircles_dino
        self._model_yolo_twocircles_sam = model_yolo_twocircles_sam
        self._logger = logging.getLogger(__name__)

    def eval(self, export_folder: str = None):
        with logging_redirect_tqdm():
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            export_folder = formatted_timestamp \
                if export_folder is None else export_folder
            export_path = RESULTS_PATH.joinpath(
                'multi_model_localization_eval_weakly_supervised', export_folder)
            os.makedirs(export_path, exist_ok=True)

            self._logger.info('Evaluate YOLO BBox (Expert Annotations)')
            results_yolo_expert = self.eval_yolo(
                model=self._model_yolo_expert)
            self._logger.info('Evaluate YOLO BBox (Cells, DINO)')
            results_yolo_cells_dino = self.eval_yolo(
                model=self._model_yolo_cells_dino)
            self._logger.info('Evaluate YOLO BBox (Cells, SAM)')
            results_yolo_cells_sam = self.eval_yolo(
                model=self._model_yolo_cells_sam)
            self._logger.info('Evaluate YOLO BBox (Two Circles, DINO)')
            results_yolo_twocircles_dino = self.eval_yolo(
                model=self._model_yolo_twocircles_dino)
            self._logger.info('Evaluate YOLO BBox (Two Circles, SAM)')
            results_yolo_twocircles_sam = self.eval_yolo(
                model=self._model_yolo_twocircles_sam)

            ious_dict = self.calculate_ious(results_yolo_cells_dino[0],
                                            results_yolo_cells_sam[0],
                                            results_yolo_expert[0],
                                            results_yolo_twocircles_dino[0],
                                            results_yolo_twocircles_sam[0],
                                            device)

            map_dict = self.calculate_map(results_yolo_expert,
                                          results_yolo_cells_dino,
                                          results_yolo_cells_sam,
                                          results_yolo_twocircles_dino,
                                          results_yolo_twocircles_sam,
                                          device)

            for entry in map_dict['maps']:
                result_dict = entry[0]

                for k, v in result_dict.items():
                    result_dict[k] = v.item()

            export_dict = {
                'intersection over union': sorted(ious_dict['ious'],
                                                  key=lambda x: x[0],
                                                  reverse=True),
                'mean average precision': sorted(map_dict['maps'],
                                                 key=lambda x: x[0]['map'],
                                                 reverse=True)
            }

            with open(export_path.joinpath('eval_metrics.json'), 'w') as f:
                json.dump(export_dict, fp=f, indent=2)

    def calculate_map(self,
                      results_yolo_expert,
                      results_yolo_cells_dino,
                      results_yolo_cells_sam,
                      results_yolo_twocircles_dino,
                      results_yolo_twocircles_sam,
                      device):
        y_test = self.get_labels()

        y_test = self.parse_labels(y_test, device)
        results_yolo_expert = self.parse_yolo_results(results_yolo_expert)
        results_yolo_cells_dino = self.parse_yolo_results(
            results_yolo_cells_dino)
        results_yolo_cells_sam = self.parse_yolo_results(
            results_yolo_cells_sam)
        results_yolo_twocircles_dino = self.parse_yolo_results(
            results_yolo_twocircles_dino)
        results_yolo_twocircles_sam = self.parse_yolo_results(
            results_yolo_twocircles_sam)

        mean_ap_metric = MeanAveragePrecision()

        predictions = [
            (results_yolo_expert, 'expert_annotations'),
            (results_yolo_cells_dino, 'cells_dino'),
            (results_yolo_cells_sam, 'cells_sam'),
            (results_yolo_twocircles_dino, 'two_circles_dino'),
            (results_yolo_twocircles_sam, 'two_circles_sam'),
        ]

        maps = []

        for prediction in predictions:
            mean_ap_metric.update(preds=prediction[0], target=y_test)
            results = mean_ap_metric.compute()
            mean_ap_metric.reset()
            maps.append((results, prediction[1]))

        maps_dict = {
            'maps': maps
        }

        return maps_dict

    def calculate_ious(self, bboxes_yolo_cells_dino,
                       bboxes_yolo_cells_sam,
                       bboxes_yolo_expert,
                       bboxes_yolo_twocircles_dino,
                       bboxes_yolo_twocircles_sam,
                       device):
        y_test = self.get_labels()

        ious_expert = self.get_ious(bboxes_yolo_expert, device, y_test)
        ious_cells_dino = self.get_ious(bboxes_yolo_cells_dino, device,
                                        y_test)
        ious_cells_sam = self.get_ious(bboxes_yolo_cells_sam, device,
                                       y_test)
        ious_twocircles_dino = self.get_ious(bboxes_yolo_twocircles_dino,
                                             device, y_test)
        ious_twocircles_sam = self.get_ious(bboxes_yolo_twocircles_sam,
                                            device, y_test)

        ious_dict = {
            'ious': [
                        (torch.mean(tensor(ious_expert,
                                           device=device)).item(),
                         'expert_annotations'),
                        (torch.mean(tensor(ious_cells_dino,
                                           device=device)).item(),
                         'cells_dino'),
                        (torch.mean(tensor(ious_cells_sam,
                                           device=device)).item(),
                         'cells_sam'),
                        (torch.mean(tensor(ious_twocircles_dino,
                                           device=device)).item(),
                         'twocircles_dino'),
                        (torch.mean(tensor(ious_twocircles_sam,
                                           device=device)).item(),
                         'twocircles_sam')
                    ]
        }

        return ious_dict

    def get_ious(self, bboxes, device, y):
        ious = []

        for entry in zip(y, bboxes):
            if entry[0].size()[0] == 0:
                ious.append(tensor(1.0, device=device))
            elif entry[1].size()[0] == 0:
                ious.append(tensor(0.0, device=device))
            else:
                iou_matrix = box_iou(entry[0], entry[1])
                ious.append(torch.mean(torch.amax(iou_matrix, dim=1)))

        return ious


class IoUModelEvaluatorUnsupervised(IoUModelEvaluator):
    def __init__(self, model_yolo_expert, data_subset, model_trainer_stfpm):
        super().__init__(model_yolo_expert, data_subset)
        self._model_stfpm = model_trainer_stfpm
        self._logger = logging.getLogger(__name__)

    def calculate_stfpm_bbox(self, maps_faulty: List[np.array], device,
                             threshold: float = 0.001) -> List[tensor]:

        # Since we train the STFPM model with an image size of 192x105 (1/10th)
        #   we need to increase the maps to get proper pixel coordinates
        masks = []
        bboxes = []
        confs = []
        maps_resized = []

        for map_faulty in maps_faulty:
            map_resized = cv2.resize(map_faulty, (1920, 1050))
            maps_resized.append(map_resized)
            masks.append(map_resized > threshold)

        contours = [cv2.findContours(x.astype(np.uint8),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE) for x in
                    masks]
        contours = [x[0] for x in contours]

        contours_cleaned = []

        for entry in contours:
            contours_cleaned.append(
                [np.vstack((z[0], z[1])).T.tolist() for z in
                 [np.squeeze(y).T for y in entry]])

        for i, entry in enumerate(contours_cleaned):
            bboxes_for_img = []
            confs_for_img = []

            for single_contour in entry:
                xy_flat_list = [x for x in
                                itertools.chain.from_iterable(single_contour)]
                x_list = xy_flat_list[0::2]
                y_list = xy_flat_list[1::2]
                x_min = min(x_list)
                x_max = max(x_list)
                y_min = min(y_list)
                y_max = max(y_list)
                bbox = [x_min, y_min, x_max, y_max]
                bboxes_for_img.append(bbox)
                confs_for_img.append(maps_resized[i][y_min:y_max + 1,
                x_min:x_max + 1].max())

            bboxes.append(tensor(bboxes_for_img, device=device))
            confs.append(tensor(confs_for_img, device=device))

        return [bboxes, confs]

    def eval(self, checkpoint_stfpm, export_folder: str = None):
        with logging_redirect_tqdm():
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            export_folder = formatted_timestamp \
                if export_folder is None else export_folder
            export_path = RESULTS_PATH.joinpath(
                'multi_model_localization_eval_unsupervised', export_folder)
            os.makedirs(export_path, exist_ok=True)

            self._logger.info('Evaluate YOLO BBox (Expert Annotations)')
            results_yolo_expert = self.eval_yolo(
                model=self._model_yolo_expert)

            self._logger.info('Evaluate STFPM')
            _, _, _, _, maps_faulty, _ = self._model_stfpm.get_eval_metrics(
                checkpoint_stfpm)

            results_stfpm_collection = []
            for threshold in [0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001,
                              0.0015, 0.0025, 0.005, 0.0075, 0.010]:
                results_stfpm = self.calculate_stfpm_bbox(maps_faulty,
                                                          device=device,
                                                          threshold=threshold)
                results_stfpm_collection.append((results_stfpm, threshold))

            ious_dict = self.calculate_ious(results_stfpm_collection,
                                            results_yolo_expert[0],
                                            device)

            map_dict = self.calculate_map(results_yolo_expert,
                                          results_stfpm_collection, device)

            for entry in map_dict['maps']:
                result_dict = entry[0]

                for k, v in result_dict.items():
                    result_dict[k] = v.item()

            export_dict = {
                'intersection over union': sorted(ious_dict['ious'],
                                                  key=lambda x: x[0],
                                                  reverse=True),
                'mean average precision': sorted(map_dict['maps'],
                                                 key=lambda x: x[0]['map'],
                                                 reverse=True)
            }

            with open(export_path.joinpath('eval_metrics.json'), 'w') as f:
                json.dump(export_dict, fp=f, indent=2)

    def calculate_map(self,
                      results_yolo_expert,
                      results_stfpm_collection,
                      device):
        y_test = self.get_labels()

        y_test = self.parse_labels(y_test, device)
        results_yolo_expert = self.parse_yolo_results(results_yolo_expert)
        results_stfpm_collection = self.parse_stfpm_results(
            results_stfpm_collection, device)

        mean_ap_metric = MeanAveragePrecision()

        predictions = [
            (results_yolo_expert, 'expert_annotations'),
        ]
        predictions.extend(results_stfpm_collection)

        maps = []

        for prediction in predictions:
            mean_ap_metric.update(preds=prediction[0], target=y_test)
            results = mean_ap_metric.compute()
            mean_ap_metric.reset()
            maps.append((results, prediction[1]))

        maps_dict = {
            'maps': maps
        }

        return maps_dict

    def parse_stfpm_results(self, results_to_parse, device):

        stfpm_thresholds_parsed = []

        for stfpm_threshold in results_to_parse:
            parsed_results = []

            max_value = 0

            for entry in stfpm_threshold[0][1]:
                if entry.size()[0] > 0:
                    temp_max = entry.max().item()
                else:
                    temp_max = 0
                max_value = temp_max if temp_max > max_value else max_value

            for i in range(len(stfpm_threshold[0][0])):
                parsed_result = {
                    'boxes': stfpm_threshold[0][0][i],
                    'scores': stfpm_threshold[0][1][i] / max_value,
                    'labels': tensor([0] * len(stfpm_threshold[0][0][i]),
                                     device=device)
                }
                parsed_results.append(parsed_result)

            stfpm_thresholds_parsed.append((parsed_results,
                                            'stfpm_' + str(
                                                stfpm_threshold[1])))

        return stfpm_thresholds_parsed

    def calculate_ious(self, bboxes_stfpm_collection,
                       bboxes_yolo_expert,
                       device):
        y_test = self.get_labels()

        ious_expert = self.get_ious(bboxes_yolo_expert, device, y_test)

        ious_stfpm_collection = []
        for results_stfpm, threshold in bboxes_stfpm_collection:
            ious_stfpm = self.get_ious(results_stfpm[0], device, y_test)
            ious_stfpm_collection.append((ious_stfpm, threshold))

        ious_dict = {
            'ious': [
                        (torch.mean(tensor(ious_expert,
                                           device=device)).item(),
                         'expert_annotations'),
                    ] +
                    [
                        (torch.mean(tensor(ious_stfpm,
                                           device=device)).item(),
                         'stfpm_' + str(threshold)) for ious_stfpm, threshold
                        in ious_stfpm_collection
                    ]
        }

        return ious_dict
