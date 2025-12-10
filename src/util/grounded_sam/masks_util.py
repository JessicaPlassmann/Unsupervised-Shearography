from typing import Any, List, Dict, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image

from src.util.grounded_sam.wrapper_classes import DetectionResult


def annotate(image: Union[Image.Image, np.ndarray],
             detection_results: List[DetectionResult]) -> np.ndarray:
    """
    Annotate the image with the detection results, if any are found.

    :param image: The image to display.
    :param detection_results: The detections to plot.
    :return: The image with the plotted detections.
    """

    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax),
                      color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}',
                    (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color.tolist(), 3)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def mask_to_polygon(mask: np.ndarray, only_largest: bool = True) -> List[
    List[int]]:
    """
    Convert a mask to the corresponding polygon. The polygon is only the
    collection of the edges making up the mask.

    :param mask: The mask to reduce to a polygon.
    :param only_largest: Set to only use the polygon with the largest area,
        since the mask might not be completely connected.
    :return: The polygon generated from the mask.
    """

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    if only_largest:
        contours = max(contours, key=cv2.contourArea)
        polygon = contours.reshape(-1, 2).tolist()
    # Extract the vertices of the contour
    else:
        polygon = [contour.reshape(-1, 2).tolist() for contour in contours]

    return polygon


def polygon_to_mask(polygon: List[List[int]],
                    image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    :param polygon: List of (x, y) coordinates representing the vertices
        of the polygon.
    :param image_shape: Shape of the image (height, width) for the mask.
    :returns: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask


def load_image(image_str: str) -> Image.Image:
    """
    Load and image either via a local path or an url.

    :param image_str: The image path to load as a string.
    :return: The loaded image.
    """

    if image_str.startswith('http'):
        image = Image.open(requests.get(image_str, stream=True).raw).convert(
            'RGB')
    else:
        image = Image.open(image_str).convert('RGB')

    return image


def get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    """
    Extract the bounding boxes from a detection result.

    :param results: The detection results of an inference.
    :return: The bounding boxes in a list.
    """

    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) \
        -> List[np.ndarray]:
    """
    Refine the provided boolean mask.

    :param masks: The masks to refine.
    :param polygon_refinement: Whether to refine the mask polygon.
    :return: The refined mask.
    """

    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask, True)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


def detect(image: Image.Image, labels: List[str], detector, processor,
           threshold, text_threshold) -> List[DetectionResult]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot
    fashion.

    :param image: The image to check.
    :param labels: The labels to look for.
    :param detector: The object detector to use.
    :param processor: The object detection post processor to use.
    :param threshold: The box threshold to use by DINO.
    :param text_threshold: The text threshold to use by DINO.
    :return: The detection results.
    """

    labels = [label if label.endswith('.') else label + '.' for label in
              labels]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text = ' '.join(labels).strip()

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = detector(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )
    results = DetectionResult.prepare_results(results)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def segment(image: Image.Image, detection_results: List[DetectionResult],
            segmentor, processor, polygon_refinement: bool = False) \
        -> List[DetectionResult]:
    """
    Segment an image using the results of a previous detector.

    :param image: The image to process.
    :param detection_results: The bbox detection results of the detector.
    :param segmentor: The segmentor to use (SAM).
    :param processor: The post processor to use on the segmentor output.
    :param polygon_refinement: Whether to refine the polygon.
    :return: A list of the detection results of the segmentor.
    """

    if len(detection_results) > 0:

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        boxes = get_boxes(detection_results)
        inputs = processor(images=image, input_boxes=boxes,
                           return_tensors='pt').to(device)

        outputs = segmentor(**inputs)
        masks = processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

    else:
        detection_results = []

    return detection_results
