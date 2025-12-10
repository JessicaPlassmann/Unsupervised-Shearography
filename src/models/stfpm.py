import torch
import torchvision

from typing import List, Tuple
from torchvision.models import ResNet18_Weights

"""
STFPM code modified and taken from https://github.com/gdwang08/STFPM
Original code is under GNU GPLv3, thus same license must apply to this project
"""


class ResNet18(torch.nn.Module):
    def __init__(self, weights: ResNet18_Weights | None):
        super(ResNet18, self).__init__()
        net = torchvision.models.resnet18(weights=weights)
        # ignore the last block and fc
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        res = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res


class STFPMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._teacher = ResNet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._student = ResNet18(weights=None)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._teacher.to(device)
        self._student.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor],
    List[torch.Tensor]]:
        self._teacher.eval()

        with torch.no_grad():
            features_teacher = self._teacher(x)
        features_student = self._student(x)

        return features_teacher, features_student
