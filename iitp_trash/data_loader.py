""" data_loader.py
Replicated in the NSML leaderboard dataset, TrashData.

This file is shown for your better understanding of NSML inference system.
You cannot modify this file. Although you change some parts of this file,
it will not included in NSML inference system.
"""
import os

from nsml import IS_ON_NSML, DATASET_PATH


def feed_infer(output_file, infer_func):
    """Prediction results feeding function.

    This function feeds your prediction results to NSML inference system.

    Args:
        output_file: string. default: pred.txt
        infer_func: function.

    pred.txt (for evaluation.py) should follow the following structure:
        img_1,1,0,1,0,1,0,0,0
        img_2,0,1,0,0,1,0,0,0
        img_3,0,0,1,0,0,0,1,0
        img_4,1,0,0,0,0,1,0,0
        img_5,0,0,1,0,0,0,0,0
        ...

    """
    if IS_ON_NSML:
        root_path = os.path.join(DATASET_PATH, 'test')
    else:
        root_path = '/home/dataset/iitp_trash_proxy/test'

    predictions_str = infer_func(root_path)
    with open(output_file, 'w') as f:
        f.write("\n".join(predictions_str))

    check_file_structure(output_file)

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def test_data_loader(root_path):
    return root_path


def check_file_structure(output_file):
    """File structure check function.

    If the structure of your result file is wrong, ValueError will occur.
    """
    with open(output_file, encoding='utf-8-sig') as f:
        for idx, line in enumerate(f.readlines()):
            items = line.strip('\n').split(',')
            if len(items) != 9:
                raise ValueError('Each line should have 9 items.')
