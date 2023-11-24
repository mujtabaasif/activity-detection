import glob
import numpy as np
from utils.load_annotations import load_labels_cxyxy


def compute_bbox_stats(labels_path):
    labels = glob.glob(labels_path)
    bboxes = []
    classes = []
    for label in labels:
        for cls, bbox in load_labels_cxyxy(label):
            print(cls, bbox)
            bboxes.append(bbox)
            classes.append(cls)

    bboxes = np.array(bboxes)
    bboxes_width = bboxes[:, 2] - bboxes[:, 0]
    bboxes_height = bboxes[:, 3] - bboxes[:, 1]

    print(f'Width: {np.mean(bboxes_width)} +/- {np.std(bboxes_width)}')
    print(f'Height: {np.mean(bboxes_height)} +/- {np.std(bboxes_height)}')
    print(f'Max Width: {np.max(bboxes_width)}')
    print(f'Max Height: {np.max(bboxes_height)}')
    print(f'Min Width: {np.min(bboxes_width)}')
    print(f'Min Height: {np.min(bboxes_height)}')

    # 90% of the bboxes are within ? pixels of the mean
    print(
        f'90% of the bboxes are within {np.mean(bboxes_width) + 2 * np.std(bboxes_width)} {np.mean(bboxes_width) - 2 * np.std(bboxes_width)} pixels of the mean width')
    print(
        f'90% of the bboxes are within {np.mean(bboxes_height) + 2 * np.std(bboxes_height)} {np.mean(bboxes_height) - 2 * np.std(bboxes_height)} pixels of the mean height')


if __name__ == '__main__':
    compute_bbox_stats('activity/dev/labels/*.txt')
