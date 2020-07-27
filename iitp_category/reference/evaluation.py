# evaluation.py
import argparse

import pandas as pd
from scipy.stats import gmean
from sklearn.metrics import f1_score
import numpy as np


def read_prediction(prediction_file):
    df = pd.read_csv(prediction_file)
    return df


def read_ground_truth(ground_truth_file):
    df = pd.read_csv(ground_truth_file)
    return df


def evaluate(prediction, ground_truth):
    if set(prediction['file_name']) != set(ground_truth['file_name']):
        raise ValueError('Prediction is missing predictions for some files.')
    if len(prediction) != len(ground_truth):
        raise ValueError('Prediction and ground truth have different about of elements.')

    df = prediction.merge(ground_truth, on='file_name')
    try:
        f1_array = f1_score(df['annotation'], df['y_pred'], average=None)
        f1_mean = gmean(f1_array)
    except:
        f1_mean = 0

    return f1_mean


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
    # try:
    #     print(evaluation_metrics(config.prediction, config.test_label_path))
    # except:
    #     print(0.0)