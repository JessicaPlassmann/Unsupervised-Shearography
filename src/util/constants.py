import pathlib

"""
Constants for the project. Do not move this file without adjusting the 
PROJECT_DIR variable accordingly, else you will break the entire project. 
"""

BASE_DIR = pathlib.Path(__file__)
PROJECT_DIR = BASE_DIR.parent.parent.parent

RESOURCES_PATH = PROJECT_DIR.joinpath('resources')
SOURCE_PATH = PROJECT_DIR.joinpath('src')
SCRIPTS_PATH = PROJECT_DIR.joinpath('scripts')
RESULTS_PATH = RESOURCES_PATH.joinpath('results')
DATASET_PATH = RESOURCES_PATH.joinpath('data', 'dataset')
SAD_DATASET_PATH = DATASET_PATH.joinpath('SADD')
SADD_IMAGES_PATH = SAD_DATASET_PATH.joinpath('images')
SADD_LABELS_PATH = SAD_DATASET_PATH.joinpath('labels', 'faulty')
EXPERT_LABELS_CSV_PATH = SADD_LABELS_PATH.joinpath('expert_annotations',
                                                   'labels.csv')

### IMAGE PATHS ###

SADD_IMAGES_FAULTY_PATH = SADD_IMAGES_PATH.joinpath('faulty')
SADD_IMAGES_GOOD_CLEAN_PATH = SADD_IMAGES_PATH.joinpath('good_clean')
SADD_IMAGES_GOOD_STRIPES_PATH = SADD_IMAGES_PATH.joinpath('good_stripes')

### LABEL PATHS ###

# We only have labels for the faulty images
YOLO_GROUNDTRUTH_LABELS_PATH = SADD_LABELS_PATH.joinpath(
    'expert_annotations', 'yolo_labels')
YOLO_CELLS_DINO_LABELS_PATH = SADD_LABELS_PATH.joinpath(
    'cells_bbox_dino', 'yolo_labels')
YOLO_CELLS_SAM_LABELS_PATH = SADD_LABELS_PATH.joinpath(
    'cells_bbox_sam', 'yolo_labels')
YOLO_CELLS_SEG_LABELS_PATH = SADD_LABELS_PATH.joinpath(
    'cells_seg_sam', 'yolo_labels')
YOLO_TWOCIRCLES_DINO_LABELS_PATH = SADD_LABELS_PATH.joinpath(
    'two_circles_bbox_dino', 'yolo_labels')
YOLO_TWOCIRCLES_SAM_LABELS_PATH = SADD_LABELS_PATH.joinpath(
    'two_circles_bbox_sam', 'yolo_labels')
YOLO_TWOCIRCLES_SEG_LABELS_PATH = SADD_LABELS_PATH.joinpath(
    'two_circles_seg_sam', 'yolo_labels')
