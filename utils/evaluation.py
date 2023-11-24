import numpy as np


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def calculate_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - intersection

    intersection = intersection.astype(np.float64)
    union = union.astype(np.float64)

    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0).astype(np.float64)

    return iou


def non_max_suppression(boxes, threshold):
    """
    Apply non-maximum suppression to eliminate redundant bounding boxes.

    Args:
    - boxes (list): List of bounding boxes, each represented as [x1, y1, x2, y2].
    - threshold (float): Overlapping threshold for considering boxes as duplicates.

    Returns:
    - List of indices to keep after non-maximum suppression.
    """
    if len(boxes) == 0:
        return []

    # Convert the boxes to (x1, y1, x2, y2) format
    boxes = np.array(boxes)

    # Sort indices based on the area of the boxes in descending order
    indices = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]

    keep = []
    while len(indices) > 0:
        # Add the index with the largest area to the list of indices to keep
        keep.append(indices[0])

        # Calculate IoU (Intersection over Union) with the rest of the boxes
        iou = calculate_iou(boxes[indices[0]], boxes[indices[1:]])

        # Remove indices of boxes with high IoU
        indices = indices[1:][iou <= threshold]

    return keep

def evaluate_bounding_boxes(ground_truth, predictions):
    # Initialize counters
    true_positives = 0
    false_positives = 0

    # Evaluate predictions
    for prediction in predictions:
        pred_label, pred_x1, pred_y1, pred_x2, pred_y2 = prediction

        # Check for any overlap with ground truth bounding box
        for gt_box in ground_truth:
            gt_label, gt_x1, gt_y1, gt_x2, gt_y2 = gt_box

            # Calculate overlap (Intersection over Union - IoU)
            x_overlap = max(0, min(pred_x2, gt_x2) - max(pred_x1, gt_x1))
            y_overlap = max(0, min(pred_y2, gt_y2) - max(pred_y1, gt_y1))
            intersection = x_overlap * y_overlap

            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            union = pred_area + gt_area - intersection

            iou = intersection / union

            # Check if there is a true positive
            if iou >= 0.10 and pred_label == gt_label:
                true_positives += 1
                break  # Stop checking other ground truth boxes for this prediction

        else:
            # If no true positive is found, count as false positive
            false_positives += 1

    # Calculate false negatives
    false_negatives = len(ground_truth) - true_positives

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

    return precision, recall, f1_score