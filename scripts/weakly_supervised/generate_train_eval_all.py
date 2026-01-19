from scripts.weakly_supervised.evaluate_models.evaluate_models import \
    evaluate_models
from scripts.weakly_supervised.generate_custom_datasets. \
    generate_custom_datasets import generate_custom_datasets
from scripts.weakly_supervised.train_models.train_models import train_models


def generate_train_eval_all() -> None:
    """
    Script to bundle all functionalities of this project. Run this to generate
    the folder structure, manage the data, train the models and evaluate all.
    Only thing to do before is to copy the original dataset and the
    corresponding ground truth in the expected folder structure. For more
    details see the README.
    """

    generate_custom_datasets()
    train_models()
    evaluate_models()


if __name__ == '__main__':
    generate_train_eval_all()
