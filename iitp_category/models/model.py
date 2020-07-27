import torch
import torch.nn.functional as F
import torchvision.models as models


class Resnet50_FC2(torch.nn.Module):
    def __init__(self, n_class=9, pretrained=True):
        super(Resnet50_FC2, self).__init__()
        self.basemodel = models.resnet50(pretrained=pretrained)
        self.linear1 = torch.nn.Linear(1000, 512)
        self.linear2 = torch.nn.Linear(512, n_class)

    def forward(self, x):
        x = self.basemodel(x)
        x = F.relu(self.linear1(x))
        out = F.softmax(self.linear2(x), dim=-1)
        pred = torch.argmax(out, dim=-1)
        return out, pred


class MobileNet(torch.nn.Module):
    def __init__(self, n_class=9, pretrained=True):
        super(MobileNet, self).__init__()
        self.basemodel = models.mobilenet_v2(pretrained=pretrained)
        self.linear1 = torch.nn.Linear(1000, n_class)

    def forward(self, x):
        x = self.basemodel(x)
        out = F.softmax(self.linear1(x))
        pred = torch.argmax(out, dim=-1)
        return out, pred