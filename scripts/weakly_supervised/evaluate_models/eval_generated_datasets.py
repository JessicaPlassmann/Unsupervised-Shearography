import argparse
import glob
import json
import os
import pathlib
import torch

from typing import Dict
from datetime import datetime
from torch import tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from tqdm import tqdm

from src.util.constants import RESULTS_PATH, \
    YOLO_CELLS_SAM_LABELS_PATH, YOLO_CELLS_DINO_LABELS_PATH, \
    YOLO_TWOCIRCLES_SAM_LABELS_PATH, YOLO_TWOCIRCLES_DINO_LABELS_PATH, \
    YOLO_GROUNDTRUTH_LABELS_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_yolo_labels(path_to_dataset: pathlib.Path) -> Dict[str, tensor]:
    """
    Load and parse the YOLO labels to the [xmin, ymin, xmax, ymax] format. We
    assume that the YOLO labels folder contains three subfolders: train, val
    and test. These folders should contain .txt files in the YOLO format for
    each image of the dataset, with the .txt file having the same name as the
    image in question. Check the following link for information on the YOLO
    format.

    https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format

    :param path_to_dataset: A path to the dataset labels that should be used.
    :return: A dictionary containing the file name as keys and bboxes with the
        format [xmin, ymin, xmax, ymax] as the values.
    """

    path_to_dataset_train = path_to_dataset.joinpath('train')
    path_to_dataset_val = path_to_dataset.joinpath('val')
    path_to_dataset_test = path_to_dataset.joinpath('test')
    label_files_train = glob.glob(str(path_to_dataset_train.joinpath('*.txt')))
    label_files_val = glob.glob(str(path_to_dataset_val.joinpath('*.txt')))
    label_files_test = glob.glob(str(path_to_dataset_test.joinpath('*.txt')))
    label_files = label_files_train + label_files_val + label_files_test

    imgs_labels = {}

    # TODO this is currently fixed since we do not check the corresponding
    #   image for the size
    img_width = 1920
    img_height = 1050

    for file_path in tqdm(label_files, desc='Load and parse yolo labels'):
        with open(file_path, 'r') as f:
            img_data = f.read().split('\n')[:-1]

        img_labels = []

        for bbox in img_data:
            bbox = [float(x) for x in bbox.split(' ')[1:]]
            x_center = bbox[0] * img_width
            y_center = bbox[1] * img_height
            box_width = bbox[2] * img_width
            box_height = bbox[3] * img_height

            # x_min, y_min, x_max, y_max
            img_labels.append([x_center - box_width / 2,
                               y_center - box_height / 2,
                               x_center + box_width / 2,
                               y_center + box_height / 2])

        imgs_labels[pathlib.Path(file_path).stem] = tensor(img_labels,
                                                           device=device)

    return imgs_labels


def get_iou(prediction: Dict[str, tensor], label: Dict[str, tensor]) \
        -> float:
    """
    Calculate the IoU (intersection over union) metric for a set of predictions
    and labels bounding boxes. If the number of predictions is not equal
    to the number of ground truth for a given image, the best match will be
    used. Thus, a larger number of predictions is not punished by this metric.
    The value 1 is the highest and best value, a value of 0 the lowest score.

    :param prediction: A dictionary containing the file name of the predictions
        as keys and bboxes with the format [xmin, ymin, xmax, ymax] as the
        values.
    :param label: A dictionary containing the file name of the label as keys
        and bboxes with the format [xmin, ymin, xmax, ymax] as the values.
    :return: The float IoU metric in the range of 0 and 1.
    """

    bboxes_x = []
    bboxes_y = []

    for k in label.keys():
        bboxes_x.append(prediction[k])
        bboxes_y.append(label[k])

    ious = []

    for entry in zip(bboxes_y, bboxes_x):
        if entry[0].size()[0] == 0:
            ious.append(tensor(1.0, device=device))
        elif entry[1].size()[0] == 0:
            ious.append(tensor(0.0, device=device))
        else:
            iou_matrix = box_iou(entry[0], entry[1])
            ious.append(torch.mean(torch.amax(iou_matrix, dim=1)))

    return torch.mean(tensor(ious, device=device)).item()


def get_map(prediction: Dict[str, tensor], label: Dict[str, tensor]) \
        -> Dict[str, float]:
    """
    Calculate the mAP (mean average precision) for a set of predictions and
    labels. We utilize the torchmetrics implementation, see this:
    https://lightning.ai/docs/torchmetrics/stable/detection
    /mean_average_precision.html

    :param prediction: A dictionary containing the file name of the predictions
        as keys and bboxes with the format [xmin, ymin, xmax, ymax] as the
        values.
    :param label: A dictionary containing the file name of the label as keys
        and bboxes with the format [xmin, ymin, xmax, ymax] as the values.
    :return: A dictionary containing various metrics like mAP and mAR.
    """

    mean_ap_metric = MeanAveragePrecision()

    bboxes_x = []
    bboxes_y = []

    for k in label.keys():
        bboxes_x.append(prediction[k])
        bboxes_y.append(label[k])

    parsed_x = []
    parsed_y = []

    # This is the format torchmetrics expects
    for entry in bboxes_x:
        parsed_x.append({
            'boxes': entry,
            'scores': torch.tensor([1.] * entry.size()[0]),
            'labels': torch.tensor([0] * entry.size()[0])
        })

    # This is the format torchmetrics expects
    for entry in bboxes_y:
        parsed_y.append({
            'boxes': entry,
            'labels': torch.tensor([0] * entry.size()[0])
        })

    mean_ap_metric.update(preds=parsed_x, target=parsed_y)

    result_dict = mean_ap_metric.compute()

    for k, v in result_dict.items():
        result_dict[k] = v.item()

    return result_dict


def eval_generated_datasets(args_namespace: argparse.Namespace):
    """
    Evaluate the generated datasets via various metrics and export the results.

    :param args_namespace: The parsed arguments of the script.
    """

    formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_folder = formatted_timestamp if args_namespace.output is None \
        else args_namespace.output
    export_path = RESULTS_PATH.joinpath('datasets_eval', export_folder)
    os.makedirs(export_path, exist_ok=True)

    # Not all images have ground truth (4311 vs 4314), so we use a
    # dictionary to
    #   exclude these images later
    path_ground_truth = YOLO_GROUNDTRUTH_LABELS_PATH
    ground_truth = get_yolo_labels(path_to_dataset=path_ground_truth)

    path_cells_dino = YOLO_CELLS_DINO_LABELS_PATH
    path_cells_sam = YOLO_CELLS_SAM_LABELS_PATH
    path_twocircles_dino = YOLO_TWOCIRCLES_DINO_LABELS_PATH
    path_twocircles_sam = YOLO_TWOCIRCLES_SAM_LABELS_PATH

    labels_cells_dino = get_yolo_labels(path_to_dataset=path_cells_dino)
    labels_cells_sam = get_yolo_labels(path_to_dataset=path_cells_sam)
    labels_twocircles_dino = get_yolo_labels(
        path_to_dataset=path_twocircles_dino)
    labels_twocircles_sam = get_yolo_labels(
        path_to_dataset=path_twocircles_sam)

    iou_cells_dino = get_iou(prediction=labels_cells_dino,
                             label=ground_truth)
    iou_cells_sam = get_iou(prediction=labels_cells_sam,
                            label=ground_truth)
    iou_twocircles_dino = get_iou(prediction=labels_twocircles_dino,
                                  label=ground_truth)
    iou_twocircles_sam = get_iou(prediction=labels_twocircles_sam,
                                 label=ground_truth)

    map_cells_dino = get_map(prediction=labels_cells_dino, label=ground_truth)
    map_cells_sam = get_map(prediction=labels_cells_sam, label=ground_truth)
    map_twocircles_dino = get_map(prediction=labels_twocircles_dino,
                                  label=ground_truth)
    map_twocircles_sam = get_map(prediction=labels_twocircles_sam,
                                 label=ground_truth)

    export_dict = {
        'intersection over union': sorted([(iou_cells_dino, 'Cells, DINO'),
                                           (iou_cells_sam, 'Cells, SAM'),
                                           (iou_twocircles_dino,
                                            'Two Circles, DINO'),
                                           (iou_twocircles_sam,
                                            'Two Circles, SAM')],
                                          # Sort by highest metric
                                          key=lambda x: x[0],
                                          reverse=True),
        'mean average precision': sorted([(map_cells_dino, 'Cells, DINO'),
                                          (map_cells_sam, 'Cells, SAM'),
                                          (map_twocircles_dino,
                                           'Two Circles, DINO'),
                                          (map_twocircles_sam,
                                           'Two Circles, SAM')],
                                         # Sort by highest metric
                                         key=lambda x: x[0]['map'],
                                         reverse=True)
    }

    with open(export_path.joinpath('eval_metrics.json'), 'w') as f:
        json.dump(export_dict, fp=f, indent=2)


def get_parsed_args_eval_generated_datasets() -> argparse.Namespace:
    """
    Parse the input arguments.

    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog='Eval datasets',
        description='Evaluate the generated datasets by DINO and SAM against '
                    'the ground truth')
    parser.add_argument('-o', '--output', dest='output', default=None)

    return parser.parse_args()


def eval_generated_datasets_main() -> None:
    """
    Main method to be used in other scripts. Evaluate the generated datasets
    and export the results.
    """

    args = get_parsed_args_eval_generated_datasets()
    args.output = 'project_eval_datasets'

    eval_generated_datasets(args_namespace=args)


if __name__ == '__main__':
    eval_generated_datasets_main()
