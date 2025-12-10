import argparse
import ultralytics

from src.model_evaluator.iou_model_evaluator import \
    IoUModelEvaluatorUnsupervised
from src.model_trainer.model_trainer import ModelTrainer
from src.util.constants import RESULTS_PATH, SOURCE_PATH
from src.util.model_configs import ModelTypes


def eval_models_iou(args_namespace: argparse.Namespace) -> None:
    """
    Evaluate the predictions of the trained models using various metrics like
    IoU, mAP, mAR et cetera. The results will be exported.
    We expect the trained models to be in specific locations according to the
    README file and the previous provided scripts to train the models.
    The used metrics can only be used for models that localize the anomaly,
    thus autoencoder can not be used.

    :param args_namespace: The parsed arguments of the script.
    """
    subset = args_namespace.subset

    if subset == 'a':
        stfpm_model_type = ModelTypes.STFPM_SUBSET_A
    elif subset == 'b':
        stfpm_model_type = ModelTypes.STFPM_SUBSET_B
    else:
        raise ValueError('Invalid subset')

    ### YOLO ###
    checkpoint_yolo_expert = RESULTS_PATH.joinpath(args_namespace.project,
                                                 'expert_annotations',
                                                 'weights', 'best.pt')
    model_yolo_expert = (
        ultralytics.YOLO(SOURCE_PATH.joinpath('util', 'yolov8-p6.yaml')).
        load(checkpoint_yolo_expert))

    ### STFPM ###
    trainer_stfpm = ModelTrainer(config=stfpm_model_type.value)
    checkpoint_stfpm = RESULTS_PATH.joinpath(
        'stfpm', f'project_model_stfpm_subset_{subset}', 'best_model.pth')

    multi_model_eval = IoUModelEvaluatorUnsupervised(
        model_yolo_expert=model_yolo_expert,
        data_subset=subset,
        model_trainer_stfpm=trainer_stfpm
    )

    multi_model_eval.eval(checkpoint_stfpm=checkpoint_stfpm,
                          export_folder=args_namespace.output)


def get_parsed_args_eval_models_roc() -> argparse.Namespace:
    """
    Parse the input arguments.

    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog='Eval models',
        description='Evaluate the trained models for the localization task')
    parser.add_argument('-o', '--output', dest='output', default=None)
    parser.add_argument('-s', '--subset', dest='subset', default=None)
    parser.add_argument('-p', '--project', dest='project', default=None)

    return parser.parse_args()


def eval_models_iou_main(output_folder: str, project: str,
                         subset: str) -> None:
    """
    Main method to be used in other scripts. Evaluate the trained models and
    export the results.
    """

    args = get_parsed_args_eval_models_roc()
    args.output = output_folder
    args.project = project
    args.subset = subset

    eval_models_iou(args_namespace=args)


if __name__ == '__main__':
    eval_models_iou_main(output_folder='eval_models_iou', project='yolo',
                         subset='a')
