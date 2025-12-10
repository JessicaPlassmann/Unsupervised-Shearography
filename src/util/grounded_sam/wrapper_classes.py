import numpy as np

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

    @classmethod
    def prepare_results(cls, detection_list: List) -> List:
        # todo currently assumes batch_size = 1
        detection_dict = detection_list[0]
        results = []

        for i in range(len(detection_dict['labels'])):
            xyxy = [round(x) for x in detection_dict['boxes'][i].tolist()]

            results.append({
                'label': detection_dict['labels'][i],
                'score': detection_dict['scores'][i].item(),
                'box': {
                    'xmax': max(xyxy[0], xyxy[2]),
                    'xmin': min(xyxy[0], xyxy[2]),
                    'ymax': max(xyxy[1], xyxy[3]),
                    'ymin': min(xyxy[1], xyxy[3])
                }
            })

        return sorted(results, key=lambda x: x['score'], reverse=True)
