import argparse

from src.model_trainer.finetune_yolo import get_parsed_args_finetune_yolo, \
    finetune_yolo


def train_models(common_seed: int = None) -> None:
    """
    Train all models of the paper: AE, ConvAE, STFPM and various YOLO models.
    Make sure that the required datasets and labels are generated first and in
    the correct directories, see README.
    The provided seed enables deterministic and thus reproducable training.

    :param common_seed: An int as a common seed for all models.
    """

    args = get_parsed_args_finetune_yolo()
    args.seed = common_seed
    args.project = 'yolo_weakly_supervised'
    args.dataset = 'expert_annotations'
    finetune_yolo(args_namespace=args)

    args = get_parsed_args_finetune_yolo()
    args.seed = common_seed
    args.project = 'yolo_weakly_supervised'
    args.dataset = 'cells'
    args.model_type = 'bbox_dino'
    finetune_yolo(args_namespace=args)

    args = get_parsed_args_finetune_yolo()
    args.seed = common_seed
    args.project = 'yolo_weakly_supervised'
    args.dataset = 'cells'
    args.model_type = 'bbox_sam'
    finetune_yolo(args_namespace=args)

    args = get_parsed_args_finetune_yolo()
    args.seed = common_seed
    args.project = 'yolo_weakly_supervised'
    args.dataset = 'cells'
    args.model_type = 'seg_sam'
    finetune_yolo(args_namespace=args)

    args = get_parsed_args_finetune_yolo()
    args.seed = common_seed
    args.project = 'yolo_weakly_supervised'
    args.dataset = 'two_circles'
    args.model_type = 'bbox_dino'
    finetune_yolo(args_namespace=args)

    args = get_parsed_args_finetune_yolo()
    args.seed = common_seed
    args.project = 'yolo_weakly_supervised'
    args.dataset = 'two_circles'
    args.model_type = 'bbox_sam'
    finetune_yolo(args_namespace=args)

    args = get_parsed_args_finetune_yolo()
    args.seed = common_seed
    args.project = 'yolo_weakly_supervised'
    args.dataset = 'two_circles'
    args.model_type = 'seg_sam'
    finetune_yolo(args_namespace=args)


def get_parsed_args_train_models() -> argparse.Namespace:
    """
    Parse the input arguments.

    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog='Train all models',
        description='Train all model types from scratch')
    parser.add_argument('-s', '--seed', type=int,
                        dest='seed', default=42)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parsed_args_train_models()

    train_models(common_seed=args.seed)
