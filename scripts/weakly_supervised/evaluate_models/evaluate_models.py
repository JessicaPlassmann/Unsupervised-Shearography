from scripts.weakly_supervised.evaluate_models.eval_generated_datasets import \
    eval_generated_datasets_main
from scripts.weakly_supervised.evaluate_models.eval_models_roc import \
    eval_models_roc_main
from scripts.weakly_supervised.evaluate_models.eval_models_iou import \
    eval_models_iou_main


def evaluate_models() -> None:
    """
    Evaluate all trained models and generated datasets, exporting the results
    to various graphs and locations. For further information see the README.
    """

    eval_generated_datasets_main()

    eval_models_roc_main(output_folder='eval_models_roc_subset_a', subset='a')
    eval_models_roc_main(output_folder='eval_models_roc_subset_b', subset='b')

    eval_models_iou_main(output_folder='eval_models_iou_subset_a', subset='a')
    eval_models_iou_main(output_folder='eval_models_iou_subset_b', subset='b')


if __name__ == '__main__':
    evaluate_models()
