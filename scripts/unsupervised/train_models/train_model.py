import argparse

from src.model_trainer.model_trainer import ModelTrainer
from src.util.model_configs import ModelTypes
from src.util.set_deterministic_behavior import set_deterministic_behavior

model_dict = {
    'ae_subset_a': ModelTypes.AUTOENCODER_SUBSET_A.value,
    'ae_subset_b': ModelTypes.AUTOENCODER_SUBSET_B.value,
    'convae_subset_a': ModelTypes.AUTOENCODER2DCONV_SUBSET_A.value,
    'convae_subset_b': ModelTypes.AUTOENCODER2DCONV_SUBSET_B.value,
    'stfpm_subset_a': ModelTypes.STFPM_SUBSET_A.value,
    'stfpm_subset_b': ModelTypes.STFPM_SUBSET_B.value,
}


def train_model(args_namespace: argparse.Namespace) -> None:
    """
    Train an individual model of your choice. Models will be evaluated and
    exported as well.

    :param args_namespace: The parsed arguments of the script.
    """

    if args_namespace.seed is not None:
        set_deterministic_behavior(seed=args_namespace.seed)

    model_config = model_dict[args_namespace.model_type]

    trainer = ModelTrainer(config=model_config,
                           export_folder=args_namespace.export_folder)
    trainer.train(seed=args_namespace.seed)
    trainer.evaluate(export_folder=args_namespace.export_folder)


def get_parsed_args_train_model():
    """
    Parse the input arguments.

    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog='Train model',
        description='Train a new model from scratch')
    parser.add_argument('-m', '--model_type',
                        dest='model_type',
                        choices=['ae_subset_a', 'convae_subset_a',
                                 'stfpm_subset_a',
                                 'ae_subset_b', 'convae_subset_b',
                                 'stfpm_subset_b'])
    parser.add_argument('-e', '--export_folder',
                        dest='export_folder')
    parser.add_argument('-s', '--seed', type=int,
                        dest='seed', default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parsed_args_train_model()

    train_model(args_namespace=args)
