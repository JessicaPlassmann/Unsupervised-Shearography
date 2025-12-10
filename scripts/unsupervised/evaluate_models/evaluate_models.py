from scripts.unsupervised.evaluate_models.resnet_tsne import resnet_tsne_main
from scripts.unsupervised.evaluate_models.eval_models_roc import \
    eval_models_roc_main
from scripts.unsupervised.evaluate_models.eval_models_iou import \
    eval_models_iou_main
from src.util.constants import SADD_IMAGES_GOOD_STRIPES_PATH, \
    SADD_IMAGES_GOOD_CLEAN_PATH, SADD_IMAGES_FAULTY_PATH


def evaluate_models() -> None:
    """
    Evaluate all trained models and generated datasets, exporting the results
    to various graphs and locations. For further information see the README.
    """

    resnet_tsne_main(output_folder='project_eval_tsne_subset_a',
                     dataset_paths_good=[SADD_IMAGES_GOOD_CLEAN_PATH],
                     dataset_paths_faulty=[SADD_IMAGES_FAULTY_PATH])
    resnet_tsne_main(output_folder='project_eval_tsne_subset_b',
                     dataset_paths_good=[SADD_IMAGES_GOOD_CLEAN_PATH,
                                         SADD_IMAGES_GOOD_STRIPES_PATH],
                     dataset_paths_faulty=[SADD_IMAGES_FAULTY_PATH])

    eval_models_roc_main(output_folder='eval_models_roc_subset_a', subset='a',
                         project='yolo_unsupervised')
    eval_models_roc_main(output_folder='eval_models_roc_subset_b', subset='b',
                         project='yolo_unsupervised')

    eval_models_iou_main(output_folder='eval_models_iou_subset_a',
                         project='yolo_unsupervised', subset='a')
    eval_models_iou_main(output_folder='eval_models_iou_subset_b',
                         project='yolo_unsupervised', subset='b')


if __name__ == '__main__':
    evaluate_models()
