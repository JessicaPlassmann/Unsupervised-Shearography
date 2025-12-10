import random
import cv2
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from typing import List, Dict, Optional, Union
from PIL import Image

from src.util.grounded_sam.masks_util import annotate, mask_to_polygon
from src.util.grounded_sam.wrapper_classes import DetectionResult


def plot_detections(
        image: Union[Image.Image, np.ndarray],
        detections: List[DetectionResult], display_img
) -> None:
    """
    Plot the detections on the image and optionally export the result.

    :param image: The image to plot.
    :param detections: The detections to visualize.
    :param export_resize_factor: The ratio to resize the image before export.
    :param display_img: Set to display the image.
    :param export_path: The path to export to if export_img is true.
    """

    annotated_image = annotate(image, detections)

    if display_img:
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.show()


def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    :param num_colors: Number of random colors to generate.
    :returns: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige',
        'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
        'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod',
        'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
        'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
        'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite',
        'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey',
        'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon',
        'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon',
        'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
        'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
        'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose',
        'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod',
        'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple',
        'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver',
        'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
        'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors,
                         min(num_colors, len(named_css_colors)))


def plot_detections_plotly(
        image: np.ndarray,
        detections: List[DetectionResult],
        class_colors: Optional[Dict[str, str]] = None
) -> None:
    """
    Plot the detections and display them using plotly.

    :param image: The image to display.
    :param detections: The detections to plot.
    :param class_colors: The colors to use for plotting.
    """

    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]

    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    annotations = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygons = mask_to_polygon(mask, False)
        polygons = polygons if isinstance(polygons, list) else [polygons]

        for polygon in polygons:
            fig.add_trace(go.Scatter(
                x=[point[0] for point in polygon] + [polygon[0][0]],
                y=[point[1] for point in polygon] + [polygon[0][1]],
                mode='lines',
                line=dict(color=class_colors[idx], width=2),
                fill='toself',
                name=f'{label}: {score:.2f}'
            ))

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type='rect',
                xref='x', yref='y',
                x0=xmin, y0=ymin,
                x1=xmax, y1=ymax,
                line=dict(color=class_colors[idx])
            )
        ]
        annotation = [
            dict(
                x=(xmin + xmax) // 2, y=(ymin + ymax) // 2,
                xref='x', yref='y',
                text=f'{label}: {score:.2f}',
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [
        dict(label='None', method='relayout', args=['shapes', []])]
    button_shapes = button_shapes + [
        dict(label=f'Detection {idx + 1}', method='relayout',
             args=['shapes', shape]) for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [
        dict(label='All', method='relayout', args=['shapes', sum(shapes, [])])]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        updatemenus=[
            dict(
                type='buttons',
                direction='up',
                buttons=button_shapes
            )
        ],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    # Show plot
    fig.show()
