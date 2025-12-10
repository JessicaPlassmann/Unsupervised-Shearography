import argparse
import json
import pathlib
import ultralytics

from src.util.constants import SOURCE_PATH, RESULTS_PATH, \
    YOLO_CELLS_DINO_LABELS_PATH, YOLO_CELLS_SAM_LABELS_PATH, \
    YOLO_CELLS_SEG_LABELS_PATH, YOLO_TWOCIRCLES_DINO_LABELS_PATH, \
    YOLO_TWOCIRCLES_SAM_LABELS_PATH, YOLO_TWOCIRCLES_SEG_LABELS_PATH, \
    YOLO_GROUNDTRUTH_LABELS_PATH, DATASET_PATH
from src.util.set_deterministic_behavior import set_deterministic_behavior

model_config_dict = {
    ('cells', 'bbox_dino'): ['yolov8-p6.yaml', 'yolov8m.pt',
                             YOLO_CELLS_DINO_LABELS_PATH,
                             'cells_bbox_dino'],
    ('cells', 'bbox_sam'): ['yolov8-p6.yaml', 'yolov8m.pt',
                            YOLO_CELLS_SAM_LABELS_PATH,
                            'cells_bbox_sam'],
    ('cells', 'seg_sam'): ['yolov8m-seg-p6.yaml', 'yolov8m-seg.pt',
                           YOLO_CELLS_SEG_LABELS_PATH,
                           'cells_seg_sam'],
    ('two_circles', 'bbox_dino'): ['yolov8-p6.yaml', 'yolov8m.pt',
                                   YOLO_TWOCIRCLES_DINO_LABELS_PATH,
                                   'two_circles_bbox_dino'],
    ('two_circles', 'bbox_sam'): ['yolov8-p6.yaml', 'yolov8m.pt',
                                  YOLO_TWOCIRCLES_SAM_LABELS_PATH,
                                  'two_circles_bbox_sam'],
    ('two_circles', 'seg_sam'): ['yolov8m-seg-p6.yaml', 'yolov8m-seg.pt',
                                 YOLO_TWOCIRCLES_SEG_LABELS_PATH,
                                 'two_circles_seg_sam']
}


def finetune_yolo(args_namespace: argparse.Namespace) -> None:
    """
    Finetune a YOLOv8 m model for the task of anomaly localization. You can
    choose the type of dataset to use and the type of task (bbox, seg). The
    training information will be exported in the usual yolo formats.

    :param args_namespace: The parsed arguments of the script.
    """

    # Set freeze=12 in the train method to freeze the first 12 layers which is
    #   the backbone of yolo
    # Training seems to suggest that freezing does not make a big difference,
    #   thus we do not freeze for now
    # Val loss is a bit lower if not freezing
    # Val images look similar, maybe a bit better for freezing
    if args_namespace.seed is not None:
        deterministic = True
        set_deterministic_behavior(seed=args_namespace.seed)
    else:
        deterministic = False

    if args_namespace.dataset == 'expert_annotations':
        model_config = ['yolov8-p6.yaml', 'yolov8m.pt',
                        YOLO_GROUNDTRUTH_LABELS_PATH,
                        'expert_annotations']
    else:
        model_config = model_config_dict[(args_namespace.dataset,
                                          args_namespace.model_type)]

    labels_path = model_config[2]

    def img2label_paths_overwrite(img_paths: list[str]) -> list[str]:
        return [str(labels_path.joinpath(pathlib.Path(path).parent.name,
                                         pathlib.Path(
                                             path).stem).absolute()) + '.txt'
                for path in img_paths]

    model = ultralytics.YOLO(
        SOURCE_PATH.joinpath('util', model_config[0])).load(
        model_config[1])
    data = DATASET_PATH.joinpath('dataset.yaml')
    project = RESULTS_PATH.joinpath(args_namespace.project)

    overwrite_yolo_dataset_class(img2label_paths_overwrite)

    results = model.train(data=data, epochs=args_namespace.epochs, imgsz=640,
                          workers=0, project=project, name=model_config[3],
                          deterministic=deterministic)

    # Dump logfile dictionary to json
    log_file = {
        'seed': args_namespace.seed,
        'export_folder': model_config[3],
        'model_to_finetune': model_config[1],
        'datset': model_config[2].stem,
        'settings': model_config[0],
    }

    with open(project.joinpath(model_config[3],
                               'training_info.json'), 'w') as f:
        json.dump(log_file, f, indent=2)


def get_parsed_args_finetune_yolo() -> argparse.Namespace:
    """
    Parse the input arguments.

    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog='Finetune YOLO',
        description='Finetune a pretrained yolo model to a new dataset')
    parser.add_argument('-m', '--model_type',
                        dest='model_type',
                        choices=['bbox_dino', 'bbox_sam', 'seg_sam'])
    parser.add_argument('-d', '--dataset',
                        dest='dataset',
                        choices=['cells', 'two_circles', 'expert_annotations'])
    parser.add_argument('-e', '--epochs', type=int,
                        dest='epochs', default=50)
    parser.add_argument('-s', '--seed', type=int,
                        dest='seed', default=None)
    parser.add_argument('-p', '--project', default='yolo')

    return parser.parse_args()


def overwrite_yolo_dataset_class(func):
    # Hack so we do not have to copy/paste the images X times for X different
    #   labels; just a stupid implementation decision by ultralytics

    def get_labels(self) -> list[dict]:
        """
        Return dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their
        integrity, and prepares them for training.

        Returns:
            (list[dict]): List of label dictionaries, each containing
            information about an image and its annotations.
        """
        self.label_files = func(self.im_files)
        cache_path = pathlib.Path(self.label_files[0]).parent.with_suffix(
            ".cache")
        try:
            cache, exists = ultralytics.data.utils.load_dataset_cache_file(
                cache_path), True  # attempt to load a *.cache file
            assert (cache[
                        "version"] ==
                    ultralytics.data.dataset.DATASET_CACHE_VERSION)  #
            # matches current version
            assert cache["hash"] == ultralytics.data.utils.get_hash(
                self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError,
                ModuleNotFoundError):
            cache, exists = self.cache_labels(
                cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop(
            "results")  # found, missing, empty, corrupt, total
        if exists and ultralytics.utils.LOCAL_RANK in {-1, 0}:
            d = (f"Scanning {cache_path}... {nf} images, {nm + ne} "
                 f"backgrounds, {nc} corrupt")
            ultralytics.utils.TQDM(None, desc=self.prefix + d, total=n,
                                   initial=n)  # display results
            if cache["msgs"]:
                ultralytics.utils.LOGGER.info(
                    "\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(
                f"No valid images found in {cache_path}. Images with "
                f"incorrectly formatted labels are ignored. "
                f"{ultralytics.data.utilsHELP_URL}"
            )
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for
                   lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            ultralytics.utils.LOGGER.warning(
                f"Box and segment counts should be equal, but got len("
                f"segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will "
                f"be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment "
                "dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            ultralytics.utils.LOGGER.warning(
                f"Labels are missing or empty in {cache_path}, training may "
                f"not work correctly. {ultralytics.data.utils.HELP_URL}")
        return labels

    ultralytics.data.dataset.YOLODataset.get_labels = get_labels


if __name__ == '__main__':
    args = get_parsed_args_finetune_yolo()

    finetune_yolo(args_namespace=args)
