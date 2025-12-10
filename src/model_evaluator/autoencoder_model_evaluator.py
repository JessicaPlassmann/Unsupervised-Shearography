import json
import logging
import os
import io
import base64
import numpy as np
import pandas as pd
import torch.nn
import plotly.express as px
import plotly.io as pio

from datetime import datetime
from pathlib import PurePath
from typing import List
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm
from dash import Dash, dcc, html, Input, Output, no_update, callback

from src.model_evaluator.generic_model_evaluator import GenericModelEvaluator
from src.util.constants import RESULTS_PATH


def np_image_to_base64(im_matrix: np.array) -> str:
    """
    Encode an image to a string representation.

    :param im_matrix: An image in the form of a numpy array
    :return: The image encoded as a string used in the webapp display.
    """

    im = Image.fromarray(im_matrix)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    buffer = io.BytesIO()
    im.save(buffer, format='jpeg')
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = 'data:image/jpeg;base64, ' + encoded_image
    return im_url


class AutoencoderModelEvaluator(GenericModelEvaluator):
    def __init__(self, threshold: float):
        super().__init__()
        self._threshold = threshold

    def eval(self, model: torch.nn.Module, test_loader: List[DataLoader],
             eval_metric: torch.nn.Module, logger: logging.Logger,
             checkpoint: str, export_name: str,
             export_folder: str = None) -> None:
        use_dash = True
        run_app = False
        show_imgs = False

        checkpoint_path = PurePath(checkpoint).parent
        formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f'Evaluate Autoencoder-Model with timestamp '
                    f'{formatted_timestamp}')
        export_folder = formatted_timestamp if export_folder is None else (
            export_folder)
        export_folder_path = RESULTS_PATH.joinpath(f'{export_name}_eval',
                                                   export_folder)
        os.makedirs(export_folder_path, exist_ok=True)

        logger.info('Prepare model ...')
        logger.info(f'Checkpoint provided! Loading: {checkpoint}')

        (count_faulty, count_good, faulty_mean, good_mean, images_faulty,
         images_good, label_faulty, label_good, loss_faulty, loss_good) = (
            self.compute_metrics(checkpoint, eval_metric, logger, model,
                                 test_loader))

        with open(checkpoint_path.joinpath('training_info.json'), 'r') as f:
            model_info = json.load(f)

        logger.info('Export eval_info.json ...')

        info_file = {
            'timestamp': formatted_timestamp,
            'checkpoint': checkpoint,
            'model_info': model_info,
            'faulty': {
                'raw': loss_faulty,
                'mean': faulty_mean,
                'detection_count': count_faulty
            },
            'good': {
                'raw': loss_good,
                'mean': good_mean,
                'detection_count': count_good
            },
        }

        with open(export_folder_path.joinpath('eval_info.json'), 'w') as f:
            json.dump(info_file, f, indent=2)

        y_test = [0] * len(loss_good) + [1] * len(loss_faulty)
        y_score = loss_good + loss_faulty

        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y_test, y_score,
                                                name='ConvAutoencoder',
                                                plot_chance_level=True,
                                                ax=ax)
        ax.set_title('2-Class (Good/Faulty) PR-Curve')
        fig.savefig(export_folder_path.joinpath('pr_curve.svg'))
        if show_imgs:
            plt.show()

        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_score,
                                         name='ConvAutoencoder', ax=ax)
        ax.set_title('2-Class (Good/Faulty) ROC-Curve')
        fig.savefig(export_folder_path.joinpath('tpfp_curve.svg'))
        if show_imgs:
            plt.show()

        if use_dash:
            df = pd.DataFrame({
                'img_id': list(range(len(loss_good))) + list(
                    range(len(loss_faulty))),
                'loss': loss_good + loss_faulty,
                'labels': label_good + label_faulty,
                'color': ['good'] * len(loss_good) + ['faulty'] * len(
                    loss_faulty)
            })

            fig = px.scatter(df, x='img_id', y='loss', color='color',
                             custom_data=['labels'])
            fig.update_traces(
                hovertemplate='<br>'.join([
                    'Labels: %{customdata[0]}'
                ])
            )
            fig.add_shape(type="line", x0=0, y0=self._threshold,
                          x1=len(loss_good), y1=self._threshold,
                          line=dict(color="Green", width=3, dash='dash'),
                          showlegend=True, name='threshold')
            fig.write_html(
                export_folder_path.joinpath('scatter_testdata.html'))

            pio.write_image(fig,
                            str(export_folder_path.joinpath(
                                'scatter_testdata.svg')),
                            format='svg')

            fig.update_traces(
                hoverinfo='none',
                hovertemplate=None,
            )

            app = Dash(__name__)

            app.layout = html.Div(
                className='container',
                children=[
                    dcc.Graph(id='graph-5', figure=fig, clear_on_unhover=True),
                    dcc.Tooltip(id='graph-tooltip-5', direction='bottom'),
                ],
            )

            images = images_good + images_faulty

            @callback(
                Output('graph-tooltip-5', 'show'),
                Output('graph-tooltip-5', 'bbox'),
                Output('graph-tooltip-5', 'children'),
                Input('graph-5', 'hoverData'),
            )
            def display_hover(hoverData):
                if hoverData is None:
                    return False, no_update, no_update

                # demo only shows the first point, but other points may also
                # be available
                hover_data = hoverData['points'][0]
                bbox = hover_data['bbox']
                num = hover_data['pointNumber']
                # We check this to determine if this is the good or the faulty
                #   point; == 0 means good, == 1 means faulty, thus we
                #   display a different image
                curv_num = hover_data['curveNumber']
                num += len(loss_good) * curv_num

                im_matrix = images[num]
                im_url = np_image_to_base64(im_matrix)
                children = [
                    html.Div([
                        html.Img(
                            src=im_url,
                            style={'width': '200px', 'display': 'block',
                                   'margin': '0 auto'},
                        ),
                        html.P(hover_data['customdata'][0],
                               style={'font-weight': 'bold'})
                    ])
                ]

                return True, bbox, children

            if run_app:
                app.run(debug=True, use_reloader=False)

    def compute_metrics(self, checkpoint, eval_metric, logger, model,
                        test_loader):
        loader_good = test_loader[0]
        loader_faulty = test_loader[1]

        saved_dict = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(saved_dict)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        logger.info('Run inference ...')

        model.eval()

        loss_faulty = []
        label_faulty = []
        images_faulty = []
        loss_good = []
        label_good = []
        images_good = []

        for batch in tqdm(loader_faulty,
                          bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                          position=1, leave=True,
                          desc='Progress Faulty Imgs'):
            print("torch.cuda.memory_allocated: %fGB" % (
                        torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
            print("torch.cuda.memory_reserved: %fGB" % (
                        torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
            print("torch.cuda.max_memory_reserved: %fGB" % (
                        torch.cuda.max_memory_reserved(
                            0) / 1024 / 1024 / 1024))

            data, labels, labels_str, img_raw = batch
            data = data.to(device)
            labels = labels.to(device)
            predictions = model(data)

            loss_faulty.append(eval_metric(predictions, labels).detach().cpu())
            label_faulty.append(str(labels_str))
            images_faulty.append(torch.squeeze(img_raw).detach().cpu().numpy())

        count_faulty = torch.sum(
            torch.stack(loss_faulty) >= self._threshold).item()
        faulty_mean = torch.mean(torch.stack(loss_faulty)).item()
        loss_faulty = [x.item() for x in loss_faulty]

        for batch in tqdm(loader_good,
                          bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                          position=1, leave=True,
                          desc='Progress Good Imgs  '):
            print("torch.cuda.memory_allocated: %fGB" % (
                        torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
            print("torch.cuda.memory_reserved: %fGB" % (
                        torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
            print("torch.cuda.max_memory_reserved: %fGB" % (
                        torch.cuda.max_memory_reserved(
                            0) / 1024 / 1024 / 1024))

            data, labels, labels_str, img_raw = batch
            data = data.to(device)
            labels = labels.to(device)
            predictions = model(data)

            loss_good.append(eval_metric(predictions, labels).detach().cpu())
            label_good.append(str(labels_str))
            images_good.append(torch.squeeze(img_raw).detach().cpu().numpy())

        count_good = torch.sum(
            torch.stack(loss_good) >= self._threshold).item()
        good_mean = torch.mean(torch.stack(loss_good)).item()
        loss_good = [x.item() for x in loss_good]

        return (count_faulty, count_good, faulty_mean, good_mean,
                images_faulty, images_good, label_faulty, label_good,
                loss_faulty, loss_good)
