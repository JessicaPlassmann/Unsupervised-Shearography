import argparse
import functools
import glob
import json
import os
import pathlib
import cv2
import numpy as np

from tqdm import tqdm
from datetime import datetime

from src.util.constants import SADD_IMAGES_FAULTY_PATH, SADD_LABELS_PATH
from src.util.grounded_sam.base_util import grounded_segmentation, \
    export_labels, get_segmentor, get_detector
from src.util.grounded_sam.masks_util import polygon_to_mask
from src.util.grounded_sam.plots import plot_detections, \
    plot_detections_plotly


def extract_yolo_segmentation(args_namespace: argparse.Namespace) -> None:
    """
    Extract bboxes and segmentation masks for a given text prompt in the YOLO
    format. This script uses DINO 1.0 and SAM 1.0.

    :param args_namespace: The parsed arguments of the script.
    """

    labels = [
        'two circles'] if args_namespace.classes is None else (
        args_namespace.classes)

    labels_without_sep = [x.split('.')[0] for x in labels]
    labels_cleaned = []

    # TODO this does not cover all cases - more than two split word combos?
    for label in labels_without_sep:
        sub_labels = label.split(' ')
        labels_cleaned.append((label, sub_labels + [label]))

    labels_map = {x.split('.')[0]: y for x, y in
                  zip(labels, range(len(labels)))}

    # If we use a lot of labels, we might need to lower the threshold (from ~
    # 0.35 to 0.15)
    threshold = args_namespace.threshold
    contour_threshold = args_namespace.threshold_con
    text_threshold = args_namespace.text_threshold
    export_resize = args_namespace.export_resize
    # Restrict found instances heuristically by setting a max size of the
    # instances in relation to the total size of the image, e.g. 0.3 for 30% of
    # the image
    max_contour_size = args_namespace.max_contour_size
    refine_polygons = args_namespace.refine_polygons
    export_img = args_namespace.export_imgs
    display_img = args_namespace.display_img
    display_plotly = args_namespace.display_plotly
    use_hq = args_namespace.use_hq
    remove_small_masks = args_namespace.remove_small_masks
    formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_str = labels[0].replace(' ', '_')

    # Note: The models are large, you need a GPU with more than 16GB GPU RAM to
    #   load them properly, else inference will slow to a crawl
    # 24GB should be enough, 32GB definitely is; the smaller non-hq models are
    #   plenty for our task tough
    # As an alternative the code could be changed to first only use DINO for
    #   all images and then SAM after that, thus you only need to hold one
    #   model in RAM which should reduce load enough to run hq on weaker GPUs
    # This on the other hand would require more (non-GPU) RAM since you need to
    #   store the DINO results for all images
    if use_hq:
        detector_id = 'IDEA-Research/grounding-dino-base'
        segmentor_id = 'facebook/sam-vit-huge'
    else:
        detector_id = 'IDEA-Research/grounding-dino-tiny'
        segmentor_id = 'facebook/sam-vit-base'

    detector, processor_detector = get_detector(detector_id)
    segmentor, processor_segmentor = get_segmentor(segmentor_id)

    # TODO add file type to args
    if args_namespace.input is None:
        img_path = SADD_IMAGES_FAULTY_PATH.joinpath('**',
                                                    f'*.'
                                                    f'{args_namespace.file_extension}')
    else:
        img_path = pathlib.Path(args_namespace.input).joinpath(
            f'*.{args_namespace.file_extension}')

    SOURCE_IMAGE_PATHS = glob.glob(str(img_path))
    SOURCE_IMAGE_PATHS = sorted(SOURCE_IMAGE_PATHS)
    OUTPUT_LABEL_PATH = SADD_LABELS_PATH

    info_json = {
        'timestamp': formatted_timestamp,
        'classes': labels_map,
        'threshold': threshold,
        'text_threshold': text_threshold,
        'export_resize': export_resize,
        'refine_polygons': refine_polygons,
        'export_img': export_img,
        'display_img': display_img,
        'display_plotly': display_plotly,
        'hq': use_hq,
        'remove_small_masks': remove_small_masks,
        'contour_threshold': contour_threshold,
        'max_contour_size': max_contour_size
    }

    os.makedirs(OUTPUT_LABEL_PATH, exist_ok=True)

    with open(OUTPUT_LABEL_PATH.joinpath(f'info_{prompt_str}.json'), 'w') as f:
        json.dump(info_json, f, indent=5)

    for image_path in tqdm(SOURCE_IMAGE_PATHS):
        image_array, detections_dino_raw, detections_sam_raw = (
            grounded_segmentation(
                image=image_path,
                labels=labels,
                threshold=threshold,
                text_threshold=text_threshold,
                polygon_refinement=refine_polygons,
                detector=detector,
                segmentor=segmentor,
                processor_detector=processor_detector,
                processor_segmentor=processor_segmentor,
                max_contour_size=max_contour_size
            ))

        img_height = image_array.shape[0]
        img_width = image_array.shape[1]
        image_path = pathlib.Path(image_path)

        if remove_small_masks:
            detections_to_keep = []

            for idx, mask in enumerate(detections_sam_raw):
                mask_uint8 = mask.mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                contours = [np.squeeze(y) for y in contours]

                # Remove contours with only one point, since it does not work
                # with contourArea
                contours = [x for x in contours if x.ndim > 1]

                contours_size = [cv2.contourArea(x) for x in contours]

                elements_to_keep = []
                for index, contour_size in enumerate(contours_size):
                    if contour_size > contour_threshold:
                        elements_to_keep.append(index)

                if elements_to_keep:
                    contours = [contours[x] for x in elements_to_keep]
                    poly_masks = [
                        (polygon_to_mask(contour,
                                         mask.mask.shape) / 255).astype(
                            np.uint8) for
                        contour in contours]

                    merged_mask = functools.reduce(
                        lambda m0, m1: np.where(m1 == 0, m0, m1), poly_masks)

                    mask.mask[:, :] = 0
                    mask.mask += merged_mask
                    detections_to_keep.append(idx)

            detections_sam_raw = [detections_sam_raw[x] for x in
                                  detections_to_keep]

        # TODO hack for now to remove combinations of multiple object
        #  categories
        #  Might want to use groundingdino.util.inference.predict from GD
        #  directly in the future to avoid this
        #  Does not do anything if we are only looking for one class anyway
        for detection in detections_sam_raw:
            for entry in labels_cleaned:
                if detection.label in entry[1]:
                    detection.label = entry[0]
                    break

        for detection in detections_dino_raw:
            for entry in labels_cleaned:
                if detection.label in entry[1]:
                    detection.label = entry[0]
                    break

        masks = [x.mask for x in detections_sam_raw]
        labels_id_sam = [labels_map[x.label] for x in detections_sam_raw]
        labels_id_dino = [labels_map[x.label] for x in detections_dino_raw]

        image_name = image_path.stem
        split = image_path.parent.name
        os.makedirs(OUTPUT_LABEL_PATH, exist_ok=True)
        export_path_label = OUTPUT_LABEL_PATH
        os.makedirs(export_path_label, exist_ok=True)
        os.makedirs(
            export_path_label.joinpath(f'{prompt_str}_seg_sam', 'yolo_labels',
                                       split), exist_ok=True)
        os.makedirs(
            export_path_label.joinpath(f'{prompt_str}_bbox_sam', 'yolo_labels',
                                       split), exist_ok=True)
        os.makedirs(
            export_path_label.joinpath(f'{prompt_str}_bbox_dino',
                                       'yolo_labels', split), exist_ok=True)

        export_labels(masks, labels_id_sam, detections_dino_raw, prompt_str,
                      split, image_name, export_path_label, img_width,
                      img_height)
        plot_detections(image_array, detections_sam_raw, display_img)

        if display_plotly:
            plot_detections_plotly(image_array, detections_sam_raw)


def get_parsed_args_extract_yolo_segmentation():
    """
    Parse the input arguments.

    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog='YOLO Segmentation Extraction',
        description='Generate Segmentation Annotations for YOLO '
                    'of Images using SAM')
    parser.add_argument('-c', '--classes', dest='classes', nargs='+',
                        default=None)
    parser.add_argument('-t', '--threshold', dest='threshold', default=0.15,
                        type=float)
    parser.add_argument('-m', '--max_contour_size', dest='max_contour_size',
                        default=0.2, type=float)
    parser.add_argument('-l', '--text_threshold', dest='text_threshold',
                        default=0.15, type=float)
    parser.add_argument('-x', '--export_resize', dest='export_resize',
                        default=0.25, type=float)
    parser.add_argument('-a', '--threshold_con', dest='threshold_con',
                        default=500, type=int)
    parser.add_argument('-i', '--input', dest='input', default=None)
    parser.add_argument('-e', '--export_imgs', dest='export_imgs',
                        default=False,
                        action='store_true')
    parser.add_argument('-r', '--refine_polygons', dest='refine_polygons',
                        default=False,
                        action='store_true')
    parser.add_argument('-d', '--display_img', dest='display_img',
                        default=False,
                        action='store_true')
    parser.add_argument('-s', '--remove_small_masks',
                        dest='remove_small_masks',
                        default=True, action='store_true')
    parser.add_argument('-p', '--display_plotly', dest='display_plotly',
                        default=False, action='store_true')
    parser.add_argument('-u', '--use_hq', dest='use_hq',
                        default=False,
                        action='store_true')
    parser.add_argument('-f', '--file_extension', dest='file_extension',
                        default='png')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parsed_args_extract_yolo_segmentation()

    extract_yolo_segmentation(args_namespace=args)
