""" data_loader.py
Replicated in the NSML leaderboard dataset.
This file is shown for your better understanding of NSML inference system.
You cannot modify this file. Although you change some parts of this file,
it will not included in NSML inference system.
"""

from nsml import IS_ON_NSML, DATASET_PATH

import os


def feed_infer(output_file, infer_func):
    """"
    infer_func(function): inference 할 유저의 함수
    output_file(str): inference 후 결과값을 저장할 파일의 위치 패스
     (이위치에 결과를 저장해야 evaluation.py 에 올바른 인자로 들어옵니다.)
    """
    if IS_ON_NSML:
        root_path = os.path.join(DATASET_PATH, 'test')
    else:
        root_path = '/home/data/iitp_falling/test'
    file_names, results = infer_func(root_path)

    # assert(len(set(file_names))==len(results))
    assert (results.shape[1] == 4)

    results_str = [fn + ',' + ','.join(str(v) for v in l) for fn, l in zip(file_names, list(results))]
    print('write output file')
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(results_str))
    print('done!')

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def test_data_loader(root_path):
    return root_path
