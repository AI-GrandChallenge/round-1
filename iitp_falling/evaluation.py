""" evaluation.py
Replicated in the NSML leaderboard dataset.
This file is shown for your better understanding of NSML inference system.
You cannot modify this file. Although you change some parts of this file,
it will not included in NSML inference system.
"""
import argparse
import torch


def compute_iou(box_a, box_b):
    max_xy = torch.min(box_a[2:], box_b[2:])
    min_xy = torch.max(box_a[:2], box_b[:2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    intersection = inter[0] * inter[1]

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union


def read_prediction_gt(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    data = [l.replace('\n', '').split(',') for l in lines]
    image_names = []
    boxes_t = []
    for d in data:
        img_name = d[0]
        box = d[1:]
        image_names.append(img_name)
        boxes_t.append(torch.Tensor([float(b) for b in box]).unsqueeze(0))
    boxes_t = torch.cat(boxes_t, 0)
    return image_names, boxes_t


def match_prediction_gt(predictions, gt_labels):
    pred_dict = dict((x, y) for x, y in zip(predictions[0], predictions[1]))
    gt_dict = dict((x, y) for x, y in zip(gt_labels[0], gt_labels[1]))
    num_preds = len(list(pred_dict.keys()))
    num_gts = len(list(gt_dict.keys()))

    prec = 0
    recl = 0
    for pred_fn in list(pred_dict.keys()):
        if pred_fn in gt_dict:
            iou = compute_iou(pred_dict[pred_fn], gt_dict[pred_fn])
            if iou > 0.5:
                prec += 1
                recl += 1

    return recl / num_gts * 100, prec / num_preds * 100


def evaluation_metrics(prediction_file, testset_path):
    predictions = read_prediction_gt(prediction_file)
    gt_labels = read_prediction_gt(testset_path)
    recall, precision = match_prediction_gt(predictions, gt_labels)
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    # return recall, precision, f1_score
    return recall


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    args.add_argument('--test_label_path', type=str)  # Ground truth  test/test_label <- csv file
    config = args.parse_args()
    try:
        print(evaluation_metrics(config.prediction, config.test_label_path))
    except:
        print('0')
