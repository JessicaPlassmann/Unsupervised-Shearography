from scripts.weakly_supervised.generate_custom_datasets.\
extract_yolo_segmentation import get_parsed_args_extract_yolo_segmentation, \
extract_yolo_segmentation


def generate_custom_datasets() -> None:
    """
    Generate the datasets used in the paper with the corresponding prompts and
    the YOLO labels (bbox and segmentation).
    """

    args = get_parsed_args_extract_yolo_segmentation()
    args.classes = ['two circles']
    extract_yolo_segmentation(args_namespace=args)

    args = get_parsed_args_extract_yolo_segmentation()
    args.classes = ['cells']
    extract_yolo_segmentation(args_namespace=args)


if __name__ == '__main__':
    generate_custom_datasets()
