# Setup
- Create a conda environment using the following command:
```
conda create -n <env_name> python=3.11
```
- Activate the environment:
```
conda activate <env_name>
```
- Install the required packages:
```
pip install -r requirements.txt
```

# Usage
- Run the script using the following command:
```
python main.py --model jozhang97/deta-swin-large --detection_save_prefix detections --extract_frames True --object_detection True --hog_features True --classifier True --predict True --evaluate True --predict_videos True --create_video True
```

# Configuration

The script can be configured by setting boolean flags at the beginning of the script. Each flag corresponds to a specific task, and users can enable or disable tasks based on their requirements.

- Extract Frames:
If EXTRACT_FRAMES is set to True, convert videos to frames using the video_to_frames function.
- Object Detection:
If OBJECT_DETECTION is set to True, perform object detection on frames using the specified model (DETECTION_MODEL). Save detection results as image labels and images.
- HOG Feature Extraction:
If HOG_FEATURES is set to True, extract HOG features from images.
- Classifier Training:
If CLASSIFIER is set to True, train a classifier using the extracted HOG features.
- Prediction:
If PREDICT is set to True, make predictions on test data using the trained classifier.
- Evaluation:
If EVALUATE is set to True, evaluate the model's performance on test data using precision, recall, and F1-score metrics.
- Predictions on Videos:
If PREDICT_VIDEOS is set to True, make predictions on video frames and save the results.
- Create Video:
If CREATE_VIDEO is set to True, create a video from the predicted frames for each video dataset.
