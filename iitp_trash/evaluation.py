""" evaluation.py
Replicated in the NSML leaderboard dataset, TrashData.

This file is shown for your better understanding of NSML inference system.
You cannot modify this file. Although you change some parts of this file,
it will not included in NSML inference system.
"""
import warnings

import argparse
import numpy as np
import pandas

from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


def evaluate(pred_csv, gt_csv):
    number_of_class = 8
    score = 0
    for i in range(number_of_class):
        pred = get_labels(pred_csv, i + 1)
        gt = get_labels(gt_csv, i + 1)
        score += f1_score(pred, gt)
    return score / 8.


def get_labels(csv_file, cid):
    return np.array(csv_file[cid])


def read_file(path):
    labels = pandas.read_csv(path, header=None).sort_values([0])
    if len(labels.columns) != 9:
        raise ValueError('Invalid number of columns.')
    return labels


def evaluation_metrics(pred_path, gt_path):
    prediction_labels = read_file(pred_path)
    gt_labels = read_file(gt_path)
    return evaluate(prediction_labels, gt_labels)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction', type=str, default='pred.txt')
    args.add_argument('--test_label_path', type=str)
    config = args.parse_args()

    try:
        print(evaluation_metrics(config.prediction, config.test_label_path))
    except:
        print(0.0)
