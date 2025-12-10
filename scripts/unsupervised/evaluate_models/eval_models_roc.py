import argparse
import ultralytics

from src.model_evaluator.roc_model_evaluator import \
    ROCModelEvaluatorUnsupervised
from src.model_trainer.model_trainer import ModelTrainer
from src.util.constants import RESULTS_PATH, SOURCE_PATH
from src.util.model_configs import ModelTypes

model_dict = {
    'a': {
        'ae': ModelTypes.AUTOENCODER_SUBSET_A,
        'convae': ModelTypes.AUTOENCODER2DCONV_SUBSET_A,
        'stfpm': ModelTypes.STFPM_SUBSET_A
    },
    'b': {
        'ae': ModelTypes.AUTOENCODER_SUBSET_B,
        'convae': ModelTypes.AUTOENCODER2DCONV_SUBSET_B,
        'stfpm': ModelTypes.STFPM_SUBSET_B
    }
}


def eval_models_roc(args_namespace: argparse.Namespace) -> None:
    """
    Evaluate the predictions of the trained models using various metrics like
    RoC, PR et cetera. The results will be exported.
    We expect the trained models to be in specific locations according to the
    README file and the previous provided scripts to train the models.

    :param args_namespace: The parsed arguments of the script.
    """

    subset = args_namespace.subset

    ### AE, ConvAE, STFPM ###
    trainer_stfpm = ModelTrainer(config=model_dict[subset]['stfpm'].value)
    trainer_ae = ModelTrainer(config=model_dict[subset]['ae'].value)
    trainer_convae = ModelTrainer(
        config=model_dict[subset]['convae'].value)
    checkpoint_ae = RESULTS_PATH.joinpath(
        'ae', f'project_model_ae_subset_{subset}', 'best_model.pth')
    checkpoint_convae = RESULTS_PATH.joinpath(
        'convae', f'project_model_convae_subset_{subset}', 'best_model.pth')
    checkpoint_stfpm = RESULTS_PATH.joinpath(
        'stfpm', f'project_model_stfpm_subset_{subset}', 'best_model.pth')

    ### YOLO ###
    checkpoint_yolo_expert = RESULTS_PATH.joinpath(args_namespace.project,
                                                   'expert_annotations',
                                                   'weights', 'best.pt')
    model_yolo_expert = (
        ultralytics.YOLO(SOURCE_PATH.joinpath('util', 'yolov8-p6.yaml')).
        load(checkpoint_yolo_expert))

    roc_model_eval = ROCModelEvaluatorUnsupervised(
        model_ae=trainer_ae,
        model_convae=trainer_convae,
        model_stfpm=trainer_stfpm,
        model_yolo_expert=model_yolo_expert,
        data_subset=subset
    )

    roc_model_eval.eval(checkpoint_ae=checkpoint_ae,
                        checkpoint_convae=checkpoint_convae,
                        checkpoint_stfpm=checkpoint_stfpm,
                        export_folder=args_namespace.output)


def get_parsed_args_eval_models_roc() -> argparse.Namespace:
    """
    Parse the input arguments.

    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog='Eval models',
        description='Evaluate the trained models for the binary task of '
                    'classifying an image into good/faulty')
    parser.add_argument('-o', '--output', dest='output', default=None)
    parser.add_argument('-s', '--subset', dest='subset', default='a')
    parser.add_argument('-p', '--project', dest='project', default='yolo')

    return parser.parse_args()


def eval_models_roc_main(output_folder: str, project: str, subset: str):
    """
    Main method to be used in other scripts. Evaluate the trained models and
    export the results.
    """

    args = get_parsed_args_eval_models_roc()
    args.output = output_folder
    args.subset = subset
    args.project = project

    eval_models_roc(args_namespace=args)


if __name__ == '__main__':
    eval_models_roc_main(output_folder='eval_models_roc_subset_a', subset='a',
                         project='yolo')
