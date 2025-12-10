import glob
import json
import logging
import os
import pathlib
import cv2
import numpy as np
import torch.nn

from datetime import datetime
from typing import List, Tuple
from matplotlib import pyplot as plt
from scipy.ndimage import binary_erosion, maximum_filter, \
    generate_binary_structure, gaussian_filter
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model_evaluator.generic_model_evaluator import GenericModelEvaluator
from src.util.constants import RESULTS_PATH

"""
STFPM code modified and taken from https://github.com/gdwang08/STFPM
Original code is under GNU GPLv3, thus same license must apply to this project
"""


def detect_peaks(image):
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = (image == 0)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def export_eval_batch(model: torch.nn.Module, eval_metric: torch.nn.Module,
                      batch: Tuple[Tuple[torch.Tensor], Tuple[str],
                      Tuple[str]], scale_factor: float,
                      img_size: Tuple[int, int],
                      export_folder: pathlib.Path, ) \
        -> Tuple[List[np.float32], List[np.float32]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    peaks = []
    means = []

    x, file_names, file_paths = batch
    x = torch.Tensor(x)
    x = x.to(device)
    features = model(x)
    maps = eval_metric(features, None)
    resized_maps = []

    for i in range(len(maps)):
        temp = cv2.resize(maps[i], img_size)
        resized_maps.append(temp)
        temp = temp * 255 * scale_factor
        heatmap_color = cv2.applyColorMap(temp.astype(np.uint8),
                                          cv2.COLORMAP_JET)

        if export_folder is not None:
            export_img_maps(export_folder, file_names, file_paths,
                            heatmap_color, i, img_size, temp)

        peaks.append(temp.max())
        means.append(temp.mean())

    return peaks, means, resized_maps


def export_img_maps(export_folder, file_names, file_paths, heatmap_color, i,
                    img_size, temp, header=''):
    threshold = 50 # 2.55?
    temp2 = (temp > threshold) * temp
    filtered_temp = gaussian_filter(temp2, 10, mode='constant')
    filtered_heatmap = cv2.applyColorMap(filtered_temp.astype(np.uint8),
                                         cv2.COLORMAP_JET)
    cv2.imwrite(str(export_folder.joinpath(header + 'heatmap_smooth_' +
                                           file_names[i])),
                filtered_heatmap)
    some_peaks = detect_peaks(filtered_temp)
    cv2.imwrite(
        str(export_folder.joinpath(header + 'peakmap_' + file_names[i])),
        some_peaks.astype(int) * 255)
    indices_peaks = np.transpose(some_peaks.nonzero())
    img_peaks = cv2.resize(cv2.imread(file_paths[i]), img_size)
    for peak in indices_peaks:
        img_peaks = cv2.circle(img_peaks, (peak[1], peak[0]), 5,
                               [0, 0, 255],
                               3)
    cv2.imwrite(str(export_folder.joinpath(header + 'peaks_' + file_names[i])),
                img_peaks)
    cv2.imwrite(
        str(export_folder.joinpath(header + 'heatmap_' + file_names[i])),
        heatmap_color)
    img = cv2.resize(cv2.imread(file_paths[i]), img_size)
    imposed_img = cv2.addWeighted(heatmap_color, 0.5, img, 0.5, 0)
    cv2.imwrite(
        str(export_folder.joinpath(header + 'superimp_' + file_names[i])),
        imposed_img)


def render_video(img_source: pathlib.Path, img_size: Tuple[int, int]) -> None:
    fps = 10
    video_name = 'superimposed'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(img_source.joinpath(video_name + '.avi')),
                            fourcc, fps, img_size)

    for img in tqdm(glob.glob(str(img_source.joinpath('superimp_*.png'))),
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    position=0, leave=True,
                    desc='Progress Video Writer'):
        img = cv2.imread(img)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


class STFPMModelEvaluator(GenericModelEvaluator):
    def __init__(self, img_size: Tuple[int, int], scale_factor: float = 1.,
                 export_video: bool = False):
        super().__init__()
        self._scale_factor = scale_factor
        self._export_video = export_video
        self._img_size = img_size

    def eval(self, model: torch.nn.Module, test_loader: List[DataLoader],
             eval_metric: torch.nn.Module, logger: logging.Logger,
             checkpoint: str, export_name: str,
             export_folder: str = None) -> None:
        show_imgs = False

        checkpoint_path = pathlib.PurePath(checkpoint).parent
        formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        export_folder = formatted_timestamp if export_folder is None else (
            export_folder)
        export_folder_path = RESULTS_PATH.joinpath(f'{export_name}_eval',
                                                   export_folder)
        logger.info(f'Evaluate STFPM-Model with timestamp '
                    f'{formatted_timestamp}')
        export_folder_faulty = export_folder_path.joinpath('faulty')
        export_folder_good = export_folder_path.joinpath('good')

        os.makedirs(export_folder_path, exist_ok=True)
        os.makedirs(export_folder_faulty, exist_ok=True)
        os.makedirs(export_folder_good, exist_ok=True)

        logger.info('Prepare model ...')
        logger.info(f'Checkpoint provided! Loading: {checkpoint}')

        means_faulty, means_good, peaks_faulty, peaks_good, _, _ = (
            self.compute_metrics(checkpoint, eval_metric, logger, model,
                                 test_loader, export_folder_faulty,
                                 export_folder_good))

        with open(checkpoint_path.joinpath('training_info.json'), 'r') as f:
            model_info = json.load(f)

        logger.info('Export eval_info.json ...')

        info_file = {
            'timestamp': formatted_timestamp,
            'export_folder': export_folder,
            'checkpoint': checkpoint,
            'model_info': model_info,
            'faulty': {
                'mean': [x.item() for x in means_faulty],
                'max': [x.item() for x in peaks_faulty]
            },
            'good': {
                'mean': [x.item() for x in means_good],
                'max': [x.item() for x in peaks_good]
            }
        }

        with open(export_folder_path.joinpath('eval_info.json'), 'w') as f:
            json.dump(info_file, f, indent=2)

        y_test = [0] * len(peaks_good) + [1] * len(peaks_faulty)
        y_score_peaks = peaks_good + peaks_faulty
        y_score_means = means_good + means_faulty

        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y_test, y_score_peaks,
                                                name='STFPM (Peaks)',
                                                plot_chance_level=True,
                                                ax=ax)
        PrecisionRecallDisplay.from_predictions(y_test, y_score_means,
                                                name='STFPM (Means)',
                                                plot_chance_level=False,
                                                ax=ax)
        ax.set_title('2-Class (Good/Faulty) PR-Curve')
        fig.savefig(export_folder_path.joinpath('pr_curve.svg'))
        if show_imgs:
            plt.show()

        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_score_peaks,
                                         name='STFPM (Peaks)', ax=ax)
        RocCurveDisplay.from_predictions(y_test, y_score_means,
                                         name='STFPM (Means)', ax=ax)
        ax.set_title('2-Class (Good/Faulty) ROC-Curve')
        fig.savefig(export_folder_path.joinpath('tpfp_curve.svg'))
        if show_imgs:
            plt.show()


        logger.info('Render superimposed videos ...')

        if self._export_video:
            render_video(export_folder_faulty, self._img_size)
            render_video(export_folder_good, self._img_size)

    def compute_metrics(self, checkpoint, eval_metric, logger, model,
                        test_loader, export_folder_faulty=None,
                        export_folder_good=None):
        loader_good = test_loader[0]
        loader_faulty = test_loader[1]

        saved_dict = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(saved_dict)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        logger.info('Run inference ...')

        model.eval()

        peaks_faulty = []
        means_faulty = []
        maps_faulty = []
        peaks_good = []
        means_good = []
        maps_good = []

        for batch in tqdm(loader_faulty,
                          bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                          position=1, leave=True,
                          desc='Progress Faulty Imgs'):
            peaks, means, maps = export_eval_batch(model, eval_metric, batch,
                                                   self._scale_factor,
                                                   self._img_size,
                                                   export_folder_faulty)
            peaks_faulty.extend(peaks)
            means_faulty.extend(means)
            maps_faulty.extend(maps)

        for batch in tqdm(loader_good,
                          bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                          position=1, leave=True,
                          desc='Progress Good Imgs  '):
            peaks, means, maps = export_eval_batch(model, eval_metric, batch,
                                                   self._scale_factor,
                                                   self._img_size,
                                                   export_folder_good)
            peaks_good.extend(peaks)
            means_good.extend(means)
            maps_good.extend(maps)

        return (means_faulty, means_good, peaks_faulty, peaks_good,
                maps_faulty, maps_good)
