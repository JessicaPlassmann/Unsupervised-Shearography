import argparse
import base64
import glob
import io
import os
import pathlib
from typing import Any, List

import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.io as pio

from dash import Dash, dcc, html, Input, Output, no_update, callback
from datetime import datetime
from PIL import Image
from dash._callback import NoUpdate
from dash.html import Div
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from tqdm import tqdm

from src.util.constants import RESULTS_PATH, SADD_IMAGES_GOOD_STRIPES_PATH, \
    SADD_IMAGES_FAULTY_PATH, SADD_IMAGES_GOOD_CLEAN_PATH


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


def resnet_tsne(args_namespace: argparse.Namespace,
                dataset_paths_good: List[pathlib.Path],
                dataset_paths_faulty: List[pathlib.Path]) -> None:
    """
    Evaluate the ground truth dataset by extracting some features via ResNet
    and reduce these features via PCA and tSNE to check for separation. The
    results will be plotted and exported.

    :param args_namespace: The parsed arguments of the script.
    """

    show_imgs = False
    use_app = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_good = []
    x_faulty = []

    imgs_paths_good = []
    imgs_paths_faulty = []

    for dataset_path in dataset_paths_good:
        imgs_paths_good += glob.glob(
            str(dataset_path.joinpath('test', '*.png')))

    for dataset_path in dataset_paths_faulty:
        imgs_paths_faulty += glob.glob(
            str(dataset_path.joinpath('test', '*.png')))

    imgs_good = []
    imgs_faulty = []

    for entry in tqdm(imgs_paths_good):
        img = Image.open(entry).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        x_good.append(preprocess(img).to(device))
        imgs_good.append(np.array(img))

    for entry in tqdm(imgs_paths_faulty):
        img = Image.open(entry).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        x_faulty.append(preprocess(img).to(device))
        imgs_faulty.append(np.array(img))

    x = x_good + x_faulty
    y = ['Good'] * len(x_good) + ['Faulty'] * len(x_faulty)
    x = torch.stack(x)

    model = torch.hub.load('pytorch/vision:v0.17.1', 'resnet18',
                           weights='ResNet18_Weights.DEFAULT')
    model.to(device)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    with torch.no_grad():
        output_features = feature_extractor(x)

    output_features = output_features.squeeze().cpu().numpy()

    print('Loaded. Start PCA fit ...')

    pca = PCA(n_components=0.9)
    pca.fit(output_features)

    print('Fitted. Start PCA transform ...')

    x_pca = pca.transform(output_features)

    print('Transformed.')

    n_comps = 3

    x_tsne = TSNE(n_components=n_comps, learning_rate='auto', init='pca',
                  verbose=True).fit_transform(x_pca)

    cdict = {'Good': 'green',
             'Faulty': 'red'}

    fig, ax = plt.subplots()

    for g in np.unique(y):
        ix = np.where(np.array(y) == g)
        ax.scatter(x_tsne[:, 0][ix], x_tsne[:, 1][ix], c=cdict[g], label=g,
                   s=10)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 5})
    ax.set_title('2-Class (Good/Faulty) tSNE')

    formatted_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_folder = formatted_timestamp if args_namespace.output is None \
        else args_namespace.output
    export_path = RESULTS_PATH.joinpath('tsne_eval', export_folder)
    os.makedirs(export_path, exist_ok=True)
    fig.savefig(export_path.joinpath('tsne_testdata_2d.svg'))

    if show_imgs:
        plt.show()

    data = {
        'x': x_tsne[:, 0],
        'y': x_tsne[:, 1],
        'z': x_tsne[:, 2],
        'label': y
    }

    data = pd.DataFrame(data)

    color_dict = {'Faulty': '#ff0000', 'Good': '#00ad03'}

    fig = px.scatter_3d(data, x='x', y='y', z='z',
                        color='label',
                        color_discrete_map=color_dict
                        )

    fig.update_traces(marker=dict(size=7,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.write_html(export_path.joinpath('tsne_testdata_3d.html'))
    pio.write_image(fig, str(export_path.joinpath('tsne_testdata_3d.svg')),
                    format='svg')

    if use_app:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0)
        )

        app = Dash(__name__)

        app.layout = html.Div(
            className='container',
            children=[
                dcc.Graph(id='graph-5', figure=fig, clear_on_unhover=True),
                dcc.Tooltip(id='graph-tooltip-5', direction='bottom'),
            ],
        )

        images = imgs_good + imgs_faulty

        @callback(
            Output('graph-tooltip-5', 'show'),
            Output('graph-tooltip-5', 'bbox'),
            Output('graph-tooltip-5', 'children'),
            Input('graph-5', 'hoverData'),
        )
        def display_hover(hover_data) \
                -> tuple[bool, NoUpdate, NoUpdate] | tuple[
                    bool, Any, list[Div]]:
            """
            Hover callback function for dash.

            :param hover_data: Hover input.
            :return: Callback output.
            """

            if hover_data is None:
                return False, no_update, no_update

            hover_data = hover_data['points'][0]
            bbox = hover_data['bbox']
            num = hover_data['pointNumber']
            # We check this to determine if this is the good or the faulty
            # point;
            # == 0 means good, == 1 means faulty, thus we display a
            # different image
            curv_num = hover_data['curveNumber']
            num += len(imgs_good) * curv_num

            im_matrix = images[num]
            im_url = np_image_to_base64(im_matrix)
            children = [
                html.Div([
                    html.Img(
                        src=im_url,
                        style={'width': '200px', 'display': 'block',
                               'margin': '0 auto'},
                    )
                ])
            ]

            return True, bbox, children

        app.run(debug=True, use_reloader=False)


def get_parsed_args_resnet_tsne() -> argparse.Namespace:
    """
    Parse the input arguments.

    :return: The parsed arguments.
    """

    parser = argparse.ArgumentParser(
        prog='ResNet tSNE',
        description='Generate a scatter plot with tsne of the test data')
    parser.add_argument('-o', '--output', dest='output', default=None)

    return parser.parse_args()


def resnet_tsne_main(output_folder: str,
                     dataset_paths_good: List[pathlib.Path],
                     dataset_paths_faulty: List[pathlib.Path]) -> None:
    """
    Main method to be used in other scripts. Evaluate ground truth dataset.
    """

    args = get_parsed_args_resnet_tsne()
    args.output = output_folder

    resnet_tsne(args_namespace=args,
                dataset_paths_good=dataset_paths_good,
                dataset_paths_faulty=dataset_paths_faulty)


if __name__ == '__main__':
    resnet_tsne_main('project_eval_tsne',
                     [SADD_IMAGES_GOOD_CLEAN_PATH],
                     [SADD_IMAGES_FAULTY_PATH])
