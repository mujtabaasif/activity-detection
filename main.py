import os
import glob
import argparse
import multiprocessing as mp
from imutils import paths
from utils.video_to_frames import video_to_frames, frames_to_video
from utils.load_annotations import load_labels_cxyxy
from detector import ObjectDetector
from compute_hog_features import HOG
from utils.evaluation import evaluate_bounding_boxes

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="jozhang97/deta-swin-large")
    args.add_argument("--detection_save_prefix", type=str, default="detections")
    args.add_argument("--extract_frames", type=bool, default=False)
    args.add_argument("--object_detection", type=bool, default=False)
    args.add_argument("--hog_features", type=bool, default=False)
    args.add_argument("--classifier", type=bool, default=False)
    args.add_argument("--predict", type=bool, default=False)
    args.add_argument("--evaluate", type=bool, default=False)
    args.add_argument("--predict_videos", type=bool, default=False)
    args.add_argument("--create_video", type=bool, default=True)
    args = args.parse_args()

    DETECTION_MODEL = args.model
    DETECTION_SAVE_PREFIX = args.detection_save_prefix
    EXTRACT_FRAMES = args.extract_frames
    OBJECT_DETECTION = args.object_detection
    HOG_FEATURES = args.hog_features
    CLASSIFIER = args.classifier
    PREDICT = args.predict
    EVALUATE = args.evaluate
    PREDICT_VIDEOS = args.predict_videos
    CREATE_VIDEO = args.create_video

    if EXTRACT_FRAMES:
        # Convert videos to frames
        video_to_frames('data/check/0.mp4', 'detection_results/0/frames/')
        video_to_frames('data/check/1.mp4', 'detection_results/1/frames/')
        video_to_frames('data/check/2.mp4', 'detection_results/2/frames/')

    if OBJECT_DETECTION:
        FRAMES_PATHS = ['detection_results/0/',
                        'detection_results/1/',
                        'detection_results/2/']

        # Perform Object Detection for video dataset
        for FRAMES_PATH in FRAMES_PATHS:
            # Perform Object Detection for activity dataset
            detector = ObjectDetector(model_name=DETECTION_MODEL,
                                      root_path=FRAMES_PATH,
                                      labels_prefix=f"{DETECTION_SAVE_PREFIX}_labels",
                                      images_prefix=f"{DETECTION_SAVE_PREFIX}_images")
            images = glob.glob(os.path.join(FRAMES_PATH, "frames", "*.jpg"))
            thread_pool = mp.Pool(processes=4)
            thread_pool.starmap(detector.detect, [(idx, image_path) for idx, image_path in enumerate(images)])

        # Perform Object Detection for activity dataset
        FRAMES_PATHS = ['data/activity/dev/', 'data/activity/test/']
        for FRAMES_PATH in FRAMES_PATHS:
            detector = ObjectDetector(model_name=DETECTION_MODEL,
                                      root_path=FRAMES_PATH,
                                      labels_prefix=f"{DETECTION_SAVE_PREFIX}_labels",
                                      images_prefix=f"{DETECTION_SAVE_PREFIX}_images")
            images = glob.glob(os.path.join(FRAMES_PATH, "images", "*.jpg"))
            thread_pool = mp.Pool(processes=4)
            thread_pool.starmap(detector.detect, [(idx, image_path) for idx, image_path in enumerate(images)])

    hog = HOG(orientations=18, pixelsPerCell=(11, 11), cellsPerBlock=(3, 3), transform=True)

    if HOG_FEATURES:
        # Extract HOG features
        hog.extract_descriptors('activity/dev/images/', 'activity/dev/labels/')

    hog.classifier(CLASSIFIER)

    if PREDICT:
        for idx, image_path in enumerate(sorted(list(paths.list_images('activity/test/images/')))):
            print(f"Processing {image_path} : {idx}")
            detection_path = image_path.replace('images', 'detection/labels').replace('.jpg', '.txt')
            hog.predict(image_path, detection_path)

    if EVALUATE:
        # Evaluate the model
        precision_list, recall_list, f1_score_list = list(), list(), list()
        for idx, image_path in enumerate(sorted(list(paths.list_images('activity/test/images/')))):
            print(f"Processing {image_path} : {idx}")
            detection_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
            predictions_path = image_path.replace('images', 'predictions/labels').replace('.jpg', '.txt')
            gts = [(cls, bbox[0], bbox[1], bbox[2], bbox[3]) for cls, bbox in load_labels_cxyxy(detection_path)]
            preds = [(cls, bbox[0], bbox[1], bbox[2], bbox[3]) for cls, bbox in load_labels_cxyxy(predictions_path)]

            precision, recall, f1_score = evaluate_bounding_boxes(gts, preds)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

        print(f"Precision: {sum(precision_list) / len(precision_list)}")
        print(f"Recall: {sum(recall_list) / len(recall_list)}")
        print(f"F1 Score: {sum(f1_score_list) / len(f1_score_list)}")

    if PREDICT_VIDEOS:
        FRAMES_PATHS = ['detection_results/0/',
                        'detection_results/1/',
                        'detection_results/2/']

        for FRAMES_PATH in FRAMES_PATHS:
            hog.predictions_path = os.path.join(FRAMES_PATH, 'predictions')
            os.makedirs(os.path.join(hog.predictions_path, 'labels'), exist_ok=True)
            os.makedirs(os.path.join(hog.predictions_path, 'images'), exist_ok=True)
            for idx, image_path in enumerate(sorted(list(paths.list_images(os.path.join(FRAMES_PATH, 'frames/'))))):
                print(f"Processing {image_path} : {idx}")
                detection_path = image_path.replace('frames', 'labels').replace('.jpg', '.txt')
                hog.predict(image_path, detection_path)

    if CREATE_VIDEO:
        FRAMES_PATHS = [  # 'detection_results/0/predictions/images/',
            # 'detection_results/1/predictions/images/',
            'detection_results/2/predictions/images/']

        for FRAMES_PATH in FRAMES_PATHS:
            images = sorted(list(paths.list_images(FRAMES_PATH)))
            video_name = FRAMES_PATH.split('/')[1]
            video_name = os.path.join(FRAMES_PATH.split('/')[0], f"{video_name}_predictions.mp4")
            frames_to_video(FRAMES_PATH, video_name, fps=30)
