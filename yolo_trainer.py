from ultralytics import YOLO

if __name__ == '__main__':
    # Get the predictions on all videos

    # Load a model
    model = YOLO("yolov8s.yaml")  # build a new model from scratch
    model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="configs/activity.yaml", epochs=10)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set

    model.predict('detection_results/0/frames/', save=True, save_txt=True, project='detection_results')