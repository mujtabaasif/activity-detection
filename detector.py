import os

import torch
from transformers import AutoImageProcessor, DetaForObjectDetection
from PIL import Image, ImageDraw
import glob
import multiprocessing as mp

class ObjectDetector:

    def __init__(self, model_name, root_path, labels_prefix="labels", images_prefix="images"):
        self.model_name = model_name
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DetaForObjectDetection.from_pretrained(model_name)
        self.root_path = root_path

        self.labels_prefix = labels_prefix
        self.images_prefix = images_prefix

        # Create labels directory if it does not exist
        os.makedirs(os.path.join(self.root_path, labels_prefix), exist_ok=True)
        # Create images directory if it does not exist
        os.makedirs(os.path.join(self.root_path, images_prefix), exist_ok=True)

    def detect(self, idx, image_path):
        # Get image name without extension
        print(f"Processing {image_path} : {idx}")
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.25, target_sizes=target_sizes)[0]

        # Save labels to file and save image with bounding box
        with open(os.path.join(self.root_path, self.labels_prefix, f"{image_name}.txt"), "w") as f:
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                f.write(f"{self.model.config.id2label[label.item()]} {score.item()} {' '.join([str(bb) for bb in box])}\n")

                # View the image with bounding box
                draw = ImageDraw.Draw(image)
                draw.rectangle(box, outline="red", width=3)

                # add label text
                text = f"{self.model.config.id2label[label.item()]}: {round(score.item(), 3)}"

                draw.text((box[0], box[1] - 15), text, text_size = 14, fill="red")

        image.save(os.path.join(self.root_path, self.images_prefix, f"{image_name}.jpg"))

        image.close()

    def get_predictions(self, idx, image_path):
        print(f"Processing {image_path} : {idx} / {len(images)}")
        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.25, target_sizes=target_sizes)[0]
        return image_path, results

    def save_predictions(self, image_path, results):
        # Get image name without extension
        try:
            image_name = os.path.basename(image_path).split(".")[0]
            print(f"Saving {self.root_path} / {self.labels_prefix} / {image_name}")
            # Save labels to file and save image with bounding box
            with open(os.path.join(self.root_path, self.labels_prefix, f"{image_name}.txt"), "w") as f:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    f.write(f"{self.model.config.id2label[label.item()]} {score.item()} {' '.join([str(bb) for bb in box])}\n")
        except Exception as e:
            print(f"Error saving predictions for {image_path}: {e}")

    def predict(self):
        images = glob.glob(os.path.join(self.root_path, "frames", "*.jpg"))
        images = sorted(images, key=lambda x: int(os.path.basename(x).split(".")[0]))
        for idx, image in enumerate(images):
            print(f"Processing {image} : {idx} / {len(images)}")
            self.detect(image)

if __name__ == '__main__':
    #detector = ObjectDetector(model_name="jozhang97/deta-swin-large", root_path="detection_results/0/")
    #detector.predict()

    #detector = ObjectDetector(model_name="jozhang97/deta-swin-large", root_path="detection_results/1/")
    #detector.predict()

    #detector = ObjectDetector(model_name="jozhang97/deta-swin-large", root_path="detection_results/2/")
    #detector.predict()

    detector = ObjectDetector(model_name="jozhang97/deta-swin-large", root_path='detection_results/2/', labels_prefix="labels", images_prefix="images")
    images = glob.glob(os.path.join("detection_results/2/frames", "*.jpg"))

    #predictions = [detector.detect(idx, image_path) for idx, image_path in enumerate(images)]
    # Process images in parallel in thread pool

    # 8 processes is the sweet spot for my machine
    thread_pool = mp.Pool(processes=4)
    thread_pool.starmap(detector.detect, [(idx, image_path) for idx, image_path in enumerate(images)])

