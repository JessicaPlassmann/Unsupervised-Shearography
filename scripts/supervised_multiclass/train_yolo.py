import pathlib
import shutil
import ultralytics
import yaml

from src.util.constants import RESULTS_PATH, SADD_LABELS_PATH, SAD_DATASET_PATH


def copy_labels_modify_yaml(yaml_file: str | pathlib.Path,
                            labels_folder: str | pathlib.Path):
    """
    YOLO expects the labels to be in a specific location relative to the images
    used for training. We can not avoid this without modifying the source code,
    which we want to avoid. Thus, we copy the current / correct labels needed
    for each YOLO run just before the training process to the expected
    position.
    We also add the current absolute path to the YAML file.

    E.g. to train a YOLO on the 'one_class' labels, we copy the content of
    ./resources/data/dataset/SADD/labels/faulty/export_annotations/yolo_labels
    to ./resources/data/dataset/SADD/labels/faulty.

    :param yaml_file: Path to the .yaml file to train YOLO
    :param labels_folder: Path to the folder containing the labels file
    """
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)

    with open(yaml_file, 'w') as f:
        yaml_data['path'] = str(SAD_DATASET_PATH)
        yaml.safe_dump(yaml_data, f)

    shutil.copytree(labels_folder, SADD_LABELS_PATH, dirs_exist_ok=True)


if __name__ == '__main__':
    SEED = 42

    project_path = RESULTS_PATH.joinpath('yolo_results')

    expert_annotations = SADD_LABELS_PATH.joinpath('expert_annotations')

    current_dir = pathlib.Path(__file__).parent.absolute()
    yaml_file = current_dir.joinpath('dataset.yaml')
    copy_labels_modify_yaml(yaml_file,
                            expert_annotations.joinpath('yolo_labels'))
    model = ultralytics.YOLO('yolov8m.pt')
    model.train(
        data=yaml_file, epochs=50,
        name='one_class',
        project=project_path,
        seed=SEED
    )

    yaml_file = current_dir.joinpath('dataset_defect_classes.yaml')
    copy_labels_modify_yaml(yaml_file,
                            expert_annotations.joinpath(
                                'yolo_labels_defect_classes'))
    model = ultralytics.YOLO('yolov8m.pt')
    model.train(
        data=yaml_file, epochs=50,
        name='multi_class',
        project=project_path,
        seed=SEED
    )

    yaml_file = current_dir.joinpath('dataset_defect_classes_rot_inv.yaml')
    copy_labels_modify_yaml(yaml_file,
                            expert_annotations.joinpath(
                                'yolo_labels_defect_classes_rotation_invariant'))
    model = ultralytics.YOLO('yolov8m.pt')
    model.train(
        data=yaml_file, epochs=50,
        name='multi_class_rot_inv',
        project=project_path,
        seed=SEED
    )
