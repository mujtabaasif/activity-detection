import os
import glob
from PIL import Image


def load_labels_cxyxy(labels_path):
    with open(labels_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # Convert from x1,y1,x2,y2
            cls, x1, y1, x2, y2 = line.split(' ')
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            yield cls, bbox


def convert_labels_to_yolo_format(base_path, output_path):
    labels_files = glob.glob(os.path.join(base_path, 'labels/*.txt'))
    images_path = os.path.join(base_path, 'images/')
    os.makedirs(output_path, exist_ok=True)
    for idx, label_file in enumerate(labels_files):
        filename = os.path.basename(label_file)
        outfile = os.path.join(output_path, filename)
        img = Image.open(os.path.join(images_path, filename.replace('.txt', '.jpg')))
        img_width, img_height = img.size
        # Write normalized label file

        with open(outfile, 'w') as nf:
            # Convert to YOLO format
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    # Convert from x1,y1,x2,y2 to x,y,w,h
                    cls, x1, y1, x2, y2 = line.split(' ')
                    x = (float(x1) + float(x2)) / 2 / img_width
                    y = (float(y1) + float(y2)) / 2 / img_height
                    w = (float(x2) - float(x1)) / img_width
                    h = (float(y2) - float(y1)) / img_height
                    nf.write(f'{cls} {x} {y} {w} {h}\n')

        print(f'[{idx+1}/{len(labels_files)}] {label_file} -> {outfile}')


def load_person_detection(labels_path):
    with open(labels_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                line = line.strip()
                # Convert from x1,y1,x2,y2
                cls, conf, x1, y1, x2, y2 = line.split(' ')
                if cls == 'person':
                    bbox = [float(x1), float(y1), float(x2), float(y2)]

                    yield cls, bbox
                else:
                    continue
            except Exception as e:
                print(f"Error loading labels for {labels_path}: {e} {line}")