import numpy as np
import cv2
from skimage import exposure


def crop_image(image, rect):
    # unpack the rectangle coordinates
    image_size = image.shape
    (x1, y1, x2, y2) = rect
    # ensure the proposed bounding box does not fall outside the image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_size[1], x2)
    y2 = min(image_size[0], y2)

    # crop the image using array slices -- it's a NumPy array
    cropped = image[y1:y2, x1:x2]
    # return the cropped image
    return cropped


def visualize_hog_features(cropped, hog_image):
    # visualize HOG and the original image together
    hog_image = exposure.rescale_intensity(hog_image, out_range=(0, 255))
    hog_image = hog_image.astype("uint8")
    cropped_and_hog = np.hstack((cropped, hog_image), )
    cv2.imshow("Image", cropped_and_hog)
    cv2.waitKey(0)
    cv2.destroyWindow("Image")


def add_padding_to_bbox(bbox, max_width, max_height):
    # Compute center of bbox
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    # Add padding to bbox
    bbox[0] = int(center_x - max_width)
    bbox[1] = int(center_y - max_height)
    bbox[2] = int(center_x + max_width)
    bbox[3] = int(center_y + max_height)

    return bbox
