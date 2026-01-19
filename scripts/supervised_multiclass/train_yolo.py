import ultralytics

if __name__ == '__main__':

    model = ultralytics.YOLO('yolov8m.pt')
    model.train(
        data='dataset.yaml', epochs=50,
        name='one_class',
        project='yolo_results'
    )

    model = ultralytics.YOLO('yolov8m.pt')
    model.train(
        data='dataset_defect_classes.yaml', epochs=50,
        name='multi_class',
        project='yolo_results'
    )

    model = ultralytics.YOLO('yolov8m.pt')
    model.train(
        data='dataset_defect_classes_rot_inv.yaml', epochs=50,
        name='multi_class_rot_inv',
        project='yolo_results'
    )
