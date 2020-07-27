import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=32)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=32)
        self.pool2 = nn.MaxPool1d(kernel_size=32)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=32)
        self.linear1 = nn.Linear(1249, 256)
        self.linear2 = nn.Linear(256, 41)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.squeeze()
        x = F.relu(self.linear1(x))
        out = F.softmax(self.linear2(x), dim=-1)
        pred = torch.argsort(out, descending=True, dim=0)[:, 0:2]
        pred, _ = torch.sort(pred, dim=-1)
        return out, pred