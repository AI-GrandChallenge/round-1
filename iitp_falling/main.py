import os
import datetime

import numpy as np

import time
import torch

import torchvision.models as models
import argparse

from data_loader import feed_infer
from data_local_loader import data_loader, data_loader_local
from evaluation import evaluation_metrics

import nsml
from nsml import IS_ON_NSML

if IS_ON_NSML:
    VAL_DATASET_PATH = None
else:
    VAL_DATASET_PATH = os.path.join('/home/data/iitp_2020_fallen_final/nsml_dataset/test/test_data')
    VAL_LABEL_PATH = os.path.join('/home/data/iitp_2020_fallen_final/nsml_dataset/test/test_label')

IMG_WIDTH = 960
IMG_HEIGHT = 540


class ResNet(models.ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_resnet18():
    model = ResNet(models.resnet.BasicBlock,
                   [2, 2, 2, 2],
                   num_classes=5)
    model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    return model


def _infer(model, root_path, test_loader=None):
    """Inference function for NSML infer.

    Args:
        model: Trained model instance for inference
        root_path: Set proper path for local evaluation.
        test_loader: Data loader is defined in `data_local_loader.py`.

    Returns:
        results: tuple of (image_names, outputs)
            image_names: list of file names (size: N)
                        (ex: ['aaaa.jpg', 'bbbb.jpg', ... ])
            outputs: numpy array of bounding boxes (size: N x 4)
                        (ex: [[x1,y1,x2,y2],[x1,y1,x2,y2],...])
    """
    if test_loader is None:
        # Eval using local dataset loader
        test_loader = data_loader(
            root=os.path.join(root_path, 'test_data'),
            phase='test')
    model.eval()
    outputs = []
    image_names = []
    s_t = time.time()
    for idx, (img_names, image) in enumerate(test_loader):
        image = image.cuda()
        output = model(image)

        bbox = output[:, :4]
        conf = output[:, 4]
        bbox = bbox.sigmoid()
        conf = conf.sigmoid()
        bbox_valid = bbox
        bbox_valid = bbox_valid.detach().cpu().numpy()
        img_names = np.asarray(img_names)
        img_names_valid = img_names

        # [IMPORTANT]
        # Convert bbox coords to original image scale (960 * 540)
        # Evaluation metric is computed in original image scale.
        bbox_valid[:, 0] *= IMG_WIDTH
        bbox_valid[:, 1] *= IMG_HEIGHT
        bbox_valid[:, 2] *= IMG_WIDTH
        bbox_valid[:, 3] *= IMG_HEIGHT

        outputs.append(bbox_valid.astype(np.int16))
        image_names += list(img_names_valid)

        if time.time() - s_t > 10:
            print('Infer batch {}/{}.'.format(idx + 1, len(test_loader)))

    outputs = np.concatenate(outputs, 0)
    results = (image_names, outputs)
    return results


def local_eval(model, test_loader=None, test_label_file=None):
    """Local debugging function.

    You can use this function for debugging.
    """
    prediction_file = 'pred_train.txt'
    feed_infer(prediction_file, lambda root_path: _infer(model, root_path, test_loader=test_loader))
    if not test_label_file:
        test_label_file = os.path.join(VAL_DATASET_PATH, 'test_label')
    metric_result = evaluation_metrics(
        prediction_file,
        test_label_file
    )
    print('[Eval result] recall: {:.2f}'.format(metric_result))
    return metric_result


def bind_nsml(model):
    """NSML binding function.

    This function is used for internal process in NSML. Do not change.
    """

    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


def load_weight(model, weight_file):
    """Load trained weight.

    You should put your weight file on the root directory with the name of `weight_file`.
    """
    if os.path.isfile(weight_file):
        model.load_state_dict(torch.load(weight_file)['model'])
        print('load weight from {}.'.format(weight_file))
    else:
        print('weight file {} is not exist.'.format(weight_file))
        print('=> random initialized model will be used.')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--local-eval", default=False, action='store_true')
    args.add_argument("--weight-file", type=str, default='model.pth')

    # These arguments are reserved for nsml. Do not change.
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    # model building
    model = get_resnet18()

    # load trained weight
    load_weight(model, config.weight_file)

    if config.cuda:
        model = model.cuda()

    bind_nsml(model)
    nsml.save('model')

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        val_loader = data_loader_local(root=VAL_DATASET_PATH)
        time_ = datetime.datetime.now()
        if not IS_ON_NSML and config.local_eval:
            # Local debugging block.
            start_time = time.time()
            local_eval(model, val_loader, VAL_LABEL_PATH)
            print('{} sec'.format(time.time() - start_time))
