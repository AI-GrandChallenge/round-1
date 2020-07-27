import time

import pandas as pd
import torch
from adamp import AdamP
from scipy.stats import gmean
from sklearn.metrics import f1_score
from torch import optim
from torch.utils.data import DataLoader

from configuration.config import logger, test_transform
from data_loader import TagImageInferenceDataset
from models.model import Resnet50_FC2, MobileNet


def train(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    running_loss = 0.0
    total_loss = 0.0
    correct = 0.0
    num_data = 0.0
    for i, data in enumerate(train_loader):
        start = time.time()
        x = data['image']
        xlabel = data['label']
        x = x.to(device)
        xlabel = xlabel.to(device)

        optimizer.zero_grad()  # step과 zero_grad는 쌍을 이루는 것이라고 생각하면 됨
        out = model(x)

        logit, pred = out
        loss = criterion(logit, xlabel)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()
        correct += torch.sum(pred == xlabel).item()
        num_data += xlabel.size(0)
        if i % 100 == 0:  # print every 100 mini-batches
            logger.info("epoch: {}/{} | step: {}/{} | loss: {:.4f} | time: {:.4f} sec".format(epoch, total_epochs, i,
                                                                                              len(train_loader),
                                                                                              running_loss / 2000,
                                                                                              time.time() - start))
            running_loss = 0.0

    logger.info(
        '[{}/{}]\tloss: {:.4f}\tacc: {:.4f}'.format(epoch, total_epochs, total_loss / (i + 1), correct / num_data))
    del x, xlabel
    torch.cuda.empty_cache()
    return total_loss / (i + 1), correct / num_data


def evaluate(model, test_loader, device, criterion):
    correct = 0.0
    num_data = 0.0
    total_loss = 0.0
    label = []
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['image']
            xlabel = data['label']
            x = x.to(device)
            xlabel = xlabel.to(device)
            out = model(x)

            logit, pred = out
            loss = criterion(logit, xlabel)

            correct += torch.sum(pred == xlabel).item()
            num_data += xlabel.size(0)
            total_loss += loss.item()
            label = label + xlabel.tolist()
            prediction = prediction + pred.detach().cpu().tolist()
        del x, xlabel

    torch.cuda.empty_cache()

    f1_array = f1_score(label, prediction, average=None)
    f1_mean = gmean(f1_array)
    logger.info('test loss: {loss:.4f}\ttest acc: {acc:.4f}\ttest F1: {f1:.4f}'
                .format(loss=total_loss / (i + 1), acc=correct / num_data, f1=f1_mean))
    return total_loss / (i + 1), correct / num_data, f1_mean


def inference(model, test_path: str) -> pd.DataFrame:
    testset = TagImageInferenceDataset(root_dir=f'{test_path}/test_data',
                                       transform=test_transform)

    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    y_pred = []
    filename_list = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['image']
            x = x.to(device)
            _, pred = model(x)

            filename_list += data['image_name']
            y_pred += pred.detach().cpu().tolist()

    ret = pd.DataFrame({'file_name': filename_list, 'y_pred': y_pred})
    return ret


def select_model(model_name: str, pretrain: bool, n_class: int):
    if model_name == 'resnet50':
        model = Resnet50_FC2(n_class=n_class, pretrained=pretrain)
    elif model_name == 'mobilenet_v2':
        model = MobileNet(n_class=n_class, pretrained=pretrain)
    else:
        raise NotImplementedError('Please select in [resnet50, mobilenet_v2]')
    return model


def select_optimizer(param, opt_name: str, lr: float, weight_decay: float):
    if opt_name == 'SGD':
        optimizer = optim.SGD(param, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif opt_name == 'AdamP':
        optimizer = AdamP(param, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, nesterov=True)
    else:
        raise NotImplementedError('The optimizer should be in [SGD]')
    return optimizer
