import time

import numpy as np
import pandas as pd
import torch
from adamp import AdamP
from torch import optim
from torch.utils.data import DataLoader

from configuration.config import logger
from data_loader import AudioInferenceDataset, two_hot_encode


def calculate_iou(pred, gt):
    total_iou = 0
    pred = pred.detach().cpu()
    for idx, _pred in enumerate(pred):
        _list = np.array([_pred[0], _pred[1], gt[0][idx], gt[1][idx]])
        point_list = np.sort(_list)
        iou = (point_list[2] - point_list[1]) / (point_list[3] - point_list[0])
        total_iou += iou
    return total_iou


def train(model, train_loader, optimizer, criterion, device, epoch, total_epochs, n_class=41):
    running_loss = 0.0
    total_loss = 0.0
    total_iou = 0.0
    total_num_data = 0
    for i, data in enumerate(train_loader):
        start = time.time()
        x = data['audio']
        xlabel = data['label']
        num_data = len(xlabel[0])
        xlabel_enc = two_hot_encode(xlabel[0], xlabel[1], n_dim=n_class)
        x = x.to(device)
        xlabel_enc = xlabel_enc.to(device)

        optimizer.zero_grad()  # step과 zero_grad는 쌍을 이루는 것이라고 생각하면 됨
        out, pred = model(x)

        logit = out
        loss = criterion(logit, xlabel_enc)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()
        total_iou += calculate_iou(pred, xlabel)
        total_num_data += num_data
        if i % 100 == 0:  # print every 100 mini-batches
            logger.info("epoch: {}/{} | step: {}/{} | loss: {:.4f} | time: {:.4f} sec".format(epoch, total_epochs, i,
                                                                                              len(train_loader),
                                                                                              running_loss / 2000,
                                                                                              time.time() - start))
            running_loss = 0.0

    logger.info(
        '[{}/{}]\tloss: {:.4f}\tiou: {:.4f}'.format(epoch, total_epochs, total_loss / (i + 1), total_iou / total_num_data))
    del x, xlabel
    torch.cuda.empty_cache()
    return total_loss / (i + 1), total_iou / total_num_data


def evaluate(model, test_loader, device, criterion, n_class=41):
    total_iou = 0.0
    total_num_data = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['audio']
            xlabel = data['label']
            num_data = len(xlabel[0])
            x = x.to(device)
            xlabel_enc = two_hot_encode(xlabel[0], xlabel[1], n_dim=n_class)
            xlabel_enc = xlabel_enc.to(device)
            out = model(x)

            logit, pred = out
            loss = criterion(logit, xlabel_enc)

            total_iou += calculate_iou(pred, xlabel)
            total_num_data += num_data
            total_loss += loss.item()
        del x, xlabel

    torch.cuda.empty_cache()
    logger.info('test loss: {loss:.4f}\ttest iou: {iou:.4f}'.format(loss=total_loss / (i + 1), iou=total_iou / total_num_data))
    return total_loss / (i + 1), total_iou / total_num_data


def inference(model, test_path: str) -> pd.DataFrame:
    testset = AudioInferenceDataset(root_dir=f'{test_path}/test_data')

    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    y_start = []
    y_end = []
    filename_list = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['audio']
            x = x.to(device)
            _, pred = model(x)

            filename_list += data['file_name']
            y_start += pred.detach().cpu()[:, 0].squeeze().tolist()
            y_end += pred.detach().cpu()[:, 1].squeeze().tolist()

    ret = pd.DataFrame({'file_name': filename_list, 'start': y_start, 'end': y_end})
    return ret


def select_optimizer(param, opt_name: str, lr: float, weight_decay: float):
    if opt_name == 'SGD':
        optimizer = optim.SGD(param, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt_name == 'AdamP':
        optimizer = AdamP(param, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, nesterov=True)
    else:
        raise NotImplementedError('The optimizer should be in [SGD]')
    return optimizer
