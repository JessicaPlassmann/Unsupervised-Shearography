from scripts.unsupervised.evaluate_models.evaluate_models import \
    evaluate_models
from scripts.unsupervised.train_models.train_models import train_models


def train_eval_all() -> None:
    """
    Script to bundle all functionalities of this project. Run this to generate
    the folder structure, manage the data, train the models and evaluate all.
    Only thing to do before is to copy the original dataset and the
    corresponding ground truth in the expected folder structure. For more
    details see the README.
    """

    train_models()
    evaluate_models()


if __name__ == '__main__':
    train_eval_all()
