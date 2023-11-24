# import the necessary packages
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage import feature
from imutils import paths
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2
from utils.load_annotations import load_labels_cxyxy, load_person_detection
from utils.imutils import crop_image, add_padding_to_bbox
from joblib import dump, load
from utils.evaluation import non_max_suppression

class HOG:
    def __init__(self, orientations=9, pixelsPerCell=(8, 8), cellsPerBlock=(3, 3), transform=True):
        # store the number of orientations, pixels per cell,
        # cells per block, and whether or not power law
        # compression should be applied
        self.orienations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.transform = transform
        self.labels = []
        self.descriptors = []
        self.max_width = 100
        self.max_height = 100
        self.font = ImageFont.truetype("fonts/aerial.ttf", 20)



    def describe(self, image):
        # compute HOG for the image
        (H, hogImage) = feature.hog(image, orientations=self.orienations,
                                    pixels_per_cell=self.pixelsPerCell,
                                    cells_per_block=self.cellsPerBlock,
                                    transform_sqrt=self.transform,
                                    block_norm="L1",
                                    visualize=True)

        # return the HOG features
        return H, hogImage

    def process_image(self, image_path, label_path):
        # load the image and convert it to grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for cls, bbox in load_labels_cxyxy(label_path):
            # crop the image using array slices -- it's a NumPy array
            cropped = crop_image(gray, bbox)
            # resize the image to a fixed size, ignoring the aspect
            cropped = cv2.resize(cropped, (500, 500))

            # extract Histogram of Oriented Gradients from the logo
            H, hogImage = self.describe(cropped)

            # update the data and labels
            self.descriptors.append(H)
            self.labels.append(cls)

    def extract_descriptors(self, images_path, labels_path):
        image_paths = sorted(list(paths.list_images(images_path)))
        for idx, image_path in enumerate(image_paths):
            print(f"Processing {image_path} : {idx} / {len(image_paths)}")
            self.process_image(image_path,
                               os.path.join(labels_path, os.path.basename(image_path).replace('.jpg', '.txt')))

        # dump the dataset to file
        print("[INFO] dumping features and labels to file...")
        self.descriptors = np.array(self.descriptors)
        self.labels = np.array(self.labels)
        np.save('descriptors.npy', self.descriptors)
        np.save('labels.npy', self.labels)

        return self.descriptors, self.labels

    def classifier(self, train=True):
        # load the dataset and initialize the data matrix
        print("[INFO] extracting features...")
        self.descriptors = np.load('descriptors.npy')
        self.labels = np.load('labels.npy')

        # train the model
        if train:
            print("[INFO] training model...")
            self.model = KNeighborsClassifier(n_neighbors=5) #AdaBoostClassifier(n_estimators=10, random_state=42)
            self.model.fit(self.descriptors, self.labels)
            dump(self.model, 'classifier.joblib')
        else:
            self.model = load('classifier.joblib')

    def save_predictions(self, image_path, predictions, bboxes):
        # Get image name without extension
        print(f"Processing {image_path}")
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)

        # Save labels to file and save image with bounding box
        with open(os.path.join(self.predictions_path, 'labels', f"{image_name}.txt"), "w") as f:
            for label, box in zip(predictions, bboxes):
                f.write(
                    f"{label} {' '.join([str(bb) for bb in box])}\n")

                if label == '0':
                    label = 'ceiling_rebar'
                    color = 'blue'
                if label == '1':
                    label = 'walls_formwork'
                    color = 'red'

                # View the image with bounding box
                draw = ImageDraw.Draw(image)
                draw.rectangle(box, outline=color, width=3)
                # add label text
                text = f"{label}"

                draw.text((box[0], box[1] - 15), text, font=self.font, fill=color)

        image.save(os.path.join(self.predictions_path, 'images', f"{image_name}.jpg"))

        image.close()

    def predict(self, image_path, person_detection_path):
        # load the image and convert it to grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        bboxes = []
        predictions = []
        for cls, bbox in load_person_detection(person_detection_path):
            # Increase bbox to ROI around a person
            bbox = add_padding_to_bbox(bbox, self.max_width, self.max_height)

            cropped = crop_image(gray, bbox)
            # resize the image to a fixed size, ignoring the aspect
            cropped = cv2.resize(cropped, (500, 500))

            # detect objects in the input image and clone the image
            # so that we can draw on it
            (H, hogImage) = self.describe(cropped)
            # visualized = self.visualize_features(cropped, hogImage)
            prediction = self.model.predict(H.reshape(1, -1))[0]
            predictions.append(prediction)
            bboxes.append(bbox)

        print(f"Number of predictions: {len(predictions)}")
        # Apply non-max suppression to suppress weak, overlapping bounding boxes
        keep_indices = non_max_suppression(bboxes, 0.20)
        bboxes = [bboxes[idx] for idx in keep_indices]
        predictions = [predictions[idx] for idx in keep_indices]
        print(f"Number of predictions after non-max suppression: {len(predictions)}")

        self.save_predictions(image_path, predictions, bboxes)


if __name__ == '__main__':
    hog = HOG(orientations=18, pixelsPerCell=(7, 7), cellsPerBlock=(2, 2), transform=True)

    # hog.process_image('activity/dev/images/dev_30.jpg', 'activity/dev/labels/dev_30.txt')

    # hog.extract_descriptors('activity/dev/images/', 'activity/dev/labels/')

    hog.classifier()

    for idx, image_path in enumerate(sorted(list(paths.list_images('activity/test/images/')))):
        print(f"Processing {image_path} : {idx}")
        detection_path = image_path.replace('images', 'detection/labels').replace('.jpg', '.txt')
        hog.predict(image_path, detection_path)
