import os
import math

import argparse
import nsml
import torch
import torch.nn as nn
import torchvision.models as models

from data_loader import feed_infer
from data_local_loader import data_loader

from tqdm import tqdm
from nsml import DATASET_PATH, IS_ON_NSML

if IS_ON_NSML:
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train', 'train_data')
else:
    from evaluation import evaluation_metrics
    DATASET_PATH = '/home/dataset/iitp_trash_proxy/test'


class ClsResNet(models.ResNet):
    """Model definition.

    You can use any model for the challenge. Feel free to modify this class.
    """
    def forward(self, x, extract=False):
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


def _infer(model, root_path, test_loader=None):
    """Local inference function for NSML infer.

    Args:
        model: instance. Any model is available.
        root_path: string. Automatically set by NSML.
        test_loader: instance. Data loader is defined in `data_local_loader.py`.

    Returns:
        predictions_str: list of string.
                         ['img_1,1,0,1,0,1,0,0,0', 'img_2,0,1,0,0,1,0,0,0', ...]
    """
    model.eval()

    if test_loader is None:
        test_loader = data_loader(root=os.path.join(root_path))

    list_of_fids = []
    list_of_preds = []

    for idx, (image, fid) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        fc = model(image, extract=True)
        fc = fc.detach().cpu().numpy()
        fc = 1 * (fc > 0.5)

        list_of_fids.extend(fid)
        list_of_preds.extend(fc)

    predictions_str = []
    for idx, fid in enumerate(list_of_fids):
        test_str = fid
        for pred in list_of_preds[idx]:
            test_str += ',{}'.format(pred)
        predictions_str.append(test_str)

    return predictions_str


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

    def infer(root_path, top_k=1):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


def load_weight(model):
    """Weight loading function.

    You should put your weight file on root directory. The name of weight file
    should be 'checkpoint.pth'. If there is no 'checkpoint.pth' on root directory,
    the weights will be randomly initialized.
    """
    if os.path.isfile('checkpoint.pth'):
        state_dict = torch.load('checkpoint.pth')['state_dict']
        model.load_state_dict(state_dict, strict=True)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def local_eval(model, test_loader, gt_path):
    """Local debugging function.

    You can use this function for debugging. You may need dummy gt file.

    Args:
        model: instance.
        test_loader: instance.
        gt_path: string.

    Returns:
        metric_result: float. Performance of your method.
    """
    pred_path = 'pred.txt'
    feed_infer(pred_path, lambda root_path: _infer(model=model,
                                                   root_path=root_path,
                                                   test_loader=test_loader))
    metric_result = evaluation_metrics(pred_path, gt_path)
    print('Eval result: {:.4f}'.format(metric_result))
    return metric_result


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=8)
    args.add_argument("--cuda", type=bool, default=True)

    # These three arguments are reserved for nsml. Do not change.
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()
    num_classes = config.num_classes
    cuda = config.cuda
    mode = config.mode

    model = ClsResNet(models.resnet.BasicBlock, [2, 2, 2, 2], num_classes)
    load_weight(model)

    if cuda:
        model = model.cuda()

    if IS_ON_NSML:
        # This NSML block is mandatory. Do not change.
        bind_nsml(model)
        nsml.save('checkpoint')
        if config.pause:
            nsml.paused(scope=locals())

    if mode == 'train':
        # Local debugging block. This module is not mandatory.
        # But this would be quite useful for troubleshooting.
        gt_label = os.path.join(DATASET_PATH, 'test_label')
        loader = data_loader(root=DATASET_PATH, batch_size=64)
        result = local_eval(model, loader, gt_label)
