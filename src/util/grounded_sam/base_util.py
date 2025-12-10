import itertools
import torchvision
import torch
import cv2
import numpy as np

from typing import List, Union, Tuple
from PIL import Image
from numpy import ndarray
from transformers import AutoProcessor, AutoModelForMaskGeneration, \
    AutoModelForZeroShotObjectDetection

from src.util.grounded_sam.masks_util import load_image, detect, segment
from src.util.grounded_sam.wrapper_classes import DetectionResult


def get_segmentor(segmentor_id: str):
    """
    Load the segmentor model.

    :param segmentor_id:
    :return: The loaded segmentor with the corresponding post-processor.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    segmentor = AutoModelForMaskGeneration.from_pretrained(
        segmentor_id).to(device)
    processor = AutoProcessor.from_pretrained(segmentor_id)

    return segmentor, processor


def get_detector(detector_id: str):
    """
    Load the detector model.

    :param detector_id:
    :return: The loaded detector with the corresponding post-processor.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor = AutoProcessor.from_pretrained(detector_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        detector_id).to(device)

    return detector, processor


def grounded_segmentation(image: Union[Image.Image, str], labels: List[str],
                          detector, segmentor, processor_detector,
                          processor_segmentor, threshold, text_threshold,
                          max_contour_size, polygon_refinement: bool = False) \
        -> tuple[ndarray, list[DetectionResult | list[DetectionResult]] | list[
            DetectionResult], list[DetectionResult]]:
    """
    Grounded segmentation using DINO and SAM to extract bbox and segmentation
    masks from the input image using a text prompt.

    :param image: The image to check.
    :param labels: The labels to check for, that is the text prompts.
    :param detector: The detector to use (DINO).
    :param segmentor: The segmentor to use (SAM).
    :param processor_detector: The processor for detection post-processing.
    :param processor_segmentor: The processor for segmentation post-processing.
    :param threshold: The bbox threshold for the detector.
    :param text_threshold: The text threshold for the detector.
    :param max_contour_size: The max contour to check the size of the bboxes
        extracted by the detector; bboxes that are too large are ignored /
        removed.
    :param polygon_refinement: Whether to refine the masks/polygons.
    :return: The image, DINO and SAM detection results.
    """

    if isinstance(image, str):
        image = load_image(image)

    detections_dino = detect(image, labels, detector, processor_detector,
                             threshold, text_threshold)

    max_contour_volume = image.height * image.width * max_contour_size
    detections_to_keep = []

    for detection in detections_dino:
        bbox = detection.box
        bbox_volume = (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)

        if bbox_volume <= max_contour_volume:
            detections_to_keep.append(detection)

    detections_dino = detections_to_keep

    # TODO NMS should be for each class individually
    #   Currently we only use one class, so it makes no difference
    if len(detections_dino) > 0:
        boxes, scores = parse_to_nms_format(detections_dino)
        # Use the non-maximum suppression implementation of torchvision
        nms_indices = torchvision.ops.nms(boxes, scores,
                                          iou_threshold=0.01).int()
        detections_dino = [detections_dino[x] for x in
                           nms_indices.tolist()] if len(nms_indices) > 1 else [
            detections_dino[nms_indices]]

    detections_sam = segment(image, detections_dino, segmentor,
                             processor_segmentor,
                             polygon_refinement)

    return np.array(image), detections_dino, detections_sam


def parse_to_nms_format(detections: List) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse the detection results to a format we can feed into non-maximum
    suppression.

    :param detections: The detection results.
    :return: The bounding boxes tensor and scores tensor for NMS.
    """

    # TODO Return list of tensors grouped by classes instead
    xyxy = []
    scores = []

    for detection in detections:
        xyxy.append(detection.box.xyxy)
        scores.append(detection.score)

    return torch.Tensor(xyxy), torch.Tensor(scores)


def export_labels(masks, labels_id_sam, labels_dino, prompt_str, split,
                  image_name, export_folder, img_width, img_height) -> None:
    """
    Export generated YOLO labels for segmentation and bounding boxes.

    :param masks: The masks to turn into segmentation labels.
    :param labels_id_sam: The sam labels id.
    :param labels_dino: The DINO labels
    :param image_name: The image name used for the label file name.
    :param export_folder: The export folder to use.
    :param img_width: The image width used for the YOLO format.
    :param img_height: The image height used for the YOLO format.
    """

    export_folder_seg = export_folder.joinpath(f'{prompt_str}_seg_sam',
                                               'yolo_labels', split)
    export_folder_bbox_from_sam = export_folder.joinpath(
        f'{prompt_str}_bbox_sam', 'yolo_labels', split)
    export_folder_bbox_from_dino = export_folder.joinpath(
        f'{prompt_str}_bbox_dino', 'yolo_labels', split)

    contours = [cv2.findContours(x.astype(np.uint8),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE) for x in
                masks]
    labels_sam = list(itertools.chain.from_iterable(
        [[y[0]] * y[1] for y in
         zip(labels_id_sam, [len(x[0]) for x in contours])]))
    contours = [np.squeeze(y).T for y in
                itertools.chain.from_iterable(x[0] for x in contours)]
    contours = [[y[0] / img_width, y[1] / img_height] for y in
                contours]
    contours = [np.vstack((x[0], x[1])).T.tolist() for x in
                contours]

    with open(export_folder_seg.joinpath(f'{image_name}.txt'), 'a') as f:
        for i in range(len(contours)):
            f.write(f'{labels_sam[i]} ' +
                    ' '.join([str(x) for x in itertools.chain.from_iterable(
                        contours[i])]) + '\n')

    with open(export_folder_bbox_from_sam.joinpath(f'{image_name}.txt'),
              'a') as f:
        for i in range(len(contours)):
            xy_flat_list = [x for x in
                            itertools.chain.from_iterable(contours[i])]
            x_list = xy_flat_list[0::2]
            y_list = xy_flat_list[1::2]
            x_min = min(x_list)
            x_max = max(x_list)
            y_min = min(y_list)
            y_max = max(y_list)
            bbox = [(x_max + x_min) / 2, (y_max + y_min) / 2, x_max - x_min,
                    y_max - y_min]
            f.write(
                f'{labels_sam[i]} ' + ' '.join([str(x) for x in bbox]) + '\n')

    with open(export_folder_bbox_from_dino.joinpath(f'{image_name}.txt'),
              'a') as f:
        for i, bbox in enumerate(labels_dino):
            x_min = bbox.box.xmin / img_width
            x_max = bbox.box.xmax / img_width
            y_min = bbox.box.ymin / img_height
            y_max = bbox.box.ymax / img_height
            bbox = [(x_max + x_min) / 2, (y_max + y_min) / 2, x_max - x_min,
                    y_max - y_min]
            f.write(f'{labels_id_sam[i]} ' + ' '.join(
                [str(x) for x in bbox]) + '\n')
