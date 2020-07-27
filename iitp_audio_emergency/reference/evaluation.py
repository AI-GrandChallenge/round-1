# evaluation.py
import argparse

import numpy as np
import pandas as pd


def read_prediction(prediction_file):
    df = pd.read_csv(prediction_file)
    return df


def read_ground_truth(ground_truth_file):
    df = pd.read_csv(ground_truth_file)
    return df


def distance_to_bool(df, max_dist=40):
    ret = np.full(max_dist, False, dtype=bool)
    if len(df) == 0:
        return ret
    for idx, row in df.iterrows():
        ret[row['start']:row['end']] = True
    return ret


def calculate_iou(pred_df, gt_df):
    pred = distance_to_bool(pred_df)
    gt = distance_to_bool(gt_df)
    union = pred | gt
    intersection = pred & gt
    union_sum = sum(union)
    intersection_sum = sum(intersection)
    if union_sum == 0 and intersection_sum == 0:
        return 1
    else:
        return intersection_sum / union_sum


def evaluate(prediction, ground_truth):
    file_list = ground_truth['file_name'].unique()
    total_iou = 0.0
    for filename in file_list:
        pred_df = prediction[prediction['file_name'] == filename]
        gt_df = ground_truth[ground_truth['file_name'] == filename]
        total_iou += calculate_iou(pred_df, gt_df)

    score = total_iou / len(file_list)
    return score


def evaluation_metrics(prediction_file: str, ground_truth_file: str):
    prediction = read_prediction(prediction_file)
    ground_truth = read_ground_truth(ground_truth_file)
    return evaluate(prediction, ground_truth)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # --prediction is set by file's name that contains the result of inference. (nsml internally sets)
    args.add_argument('--prediction', type=str, default='pred.txt')  # output_file from data_loader.py
    args.add_argument('--test_label_path', type=str)  # Ground truth  test/test_label <- csv file
    config = args.parse_args()
    print(evaluation_metrics(config.prediction, config.test_label_path))
