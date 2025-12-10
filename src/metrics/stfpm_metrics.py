from typing import Any, List, Tuple
import numpy as np
import torch

"""
STFPM code modified and taken from https://github.com/gdwang08/STFPM
Original code is under GNU GPLv3, thus same license must apply to this project
"""


class STFPMTrainLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def forward(self, x: Tuple[List[torch.Tensor], List[torch.Tensor]], y:
                Any) -> torch.Tensor:
        features_teacher = x[0]
        features_student = x[1]

        loss = torch.Tensor([0.])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss = loss.to(device)

        for i in range(len(features_teacher)):
            features_teacher[i] = torch.nn.functional.normalize(
                features_teacher[i], dim=1)
            features_student[i] = torch.nn.functional.normalize(
                features_student[i], dim=1)
            loss += torch.sum((features_teacher[i] - features_student[i]) ** 2,
                              1).mean()

        return loss


class STFPMValMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._metric = STFPMTestMetric()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def forward(self, x: Tuple[List[torch.Tensor], List[torch.Tensor]], y:
                Any) -> np.ndarray:
        return self._metric(x, y).mean()


class STFPMTestMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def forward(self, x: Tuple[List[torch.Tensor], List[torch.Tensor]], y:
                Any) -> np.ndarray:
        features_teacher = x[0]
        features_student = x[1]

        score_map = 1.

        for i in range(len(features_teacher)):
            features_teacher[i] = torch.nn.functional.normalize(
                features_teacher[i], dim=1)
            features_student[i] = torch.nn.functional.normalize(
                features_student[i], dim=1)
            sm = torch.sum((features_teacher[i] - features_student[i]) ** 2, 1,
                           keepdim=True)
            sm = torch.nn.functional.interpolate(sm, size=(64, 64),
                                                 mode='bilinear',
                                                 align_corners=False)
            score_map = score_map * sm

        return score_map.squeeze().cpu().data.numpy()
