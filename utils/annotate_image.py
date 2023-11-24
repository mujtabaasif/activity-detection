import os
import glob
from PIL import Image, ImageDraw
from utils.load_annotations import load_labels_cxyxy


def annotate_images(base_path, output_path):
    labels_files = glob.glob(os.path.join(base_path, 'labels/*.txt'))
    images_path = os.path.join(base_path, 'images/')
    os.makedirs(output_path, exist_ok=True)

    for idx, label_file in enumerate(labels_files):
        filename = os.path.basename(label_file)
        img = Image.open(os.path.join(images_path, filename.replace('.txt', '.jpg')))
        # Write normalized label file
        for cls, bbox in load_labels_cxyxy(label_file):
            if cls == '0':
                cls = 'ceiling_rebar'
                color = 'blue'
            elif cls == '1':
                cls = 'walls_formwork'
                color = 'red'

            draw = ImageDraw.Draw(img)
            draw.rectangle(bbox, outline=color, width=10)
            # add label text

            text = f"{cls}"
            draw.text(((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2), text, fill=color)

        img.save(os.path.join(output_path, filename.replace('.txt', '.jpg')))
        print(
            f'[{idx + 1}/{len(labels_files)}] {label_file} -> {os.path.join(output_path, filename.replace(".txt", ".jpg"))}')

if __name__ == '__main__':
    annotate_images('../activity/test/', '../activity/test/annotated_images/')