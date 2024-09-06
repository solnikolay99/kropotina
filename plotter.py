import numpy as np
import plotly.graph_objects as go
import seaborn as seaborn
from plotly.subplots import make_subplots


def _add_markers(x, y, size, color='black', name=''):
    scater = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            colorscale='Viridis',
            opacity=0.8
        )
    )
    scater.showlegend = False
    if name != '':
        scater.name = name
    return scater


def _add_line(x, y, color='black', name=''):
    scater = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color=color, width=1)
    )
    if name == '':
        scater.showlegend = False
    else:
        scater.name = name
    return scater


def plot_2d_graph(graphs: np.array, with_markers: bool = False, colors: list = [],
                  names: list = [], directions: list = ['xy', 'xz', 'yz']):
    """
    Plotting 2d graph
    """

    seaborn.set_style("darkgrid")

    traces_xy = []
    traces_xz = []
    traces_yz = []

    for i in range(len(graphs)):
        points = graphs[i]
        p_x = [point[0] for point in points]
        p_y = [point[1] for point in points]
        p_z = [point[2] for point in points]

        color = 'black' if len(colors) < i else colors[i]
        name = '' if len(names) < i else names[i]

        if with_markers:
            traces_xy.append(_add_markers(p_x, p_y, 3, color=color, name=name))
            traces_xz.append(_add_markers(p_x, p_z, 3, color=color, name=name))
            traces_yz.append(_add_markers(p_y, p_z, 3, color=color, name=name))

        traces_xy.append(_add_line(p_x, p_y, color=color, name=name))
        traces_xz.append(_add_line(p_x, p_z, color=color, name=name))
        traces_yz.append(_add_line(p_y, p_z, color=color, name=name))

    rows = 2 if len(directions) > 2 else 1
    cols = 2 if len(directions) > 1 else 1
    fig = make_subplots(rows=rows, cols=cols)
    if 'xy' in directions:
        [fig.add_trace(traces_xy_el, 1, 1) for traces_xy_el in traces_xy]
    if 'xz' in directions:
        [fig.add_trace(traces_yz_el, 1, 2) for traces_yz_el in traces_yz]
    if 'yz' in directions:
        [fig.add_trace(traces_xz_el, 2, 1) for traces_xz_el in traces_xz]
    fig.show()


def _add_markers_3d(points: list, size, color='black', name=''):
    p_x = [point[0] for point in points]
    p_y = [point[1] for point in points]
    p_z = [point[2] for point in points]

    return go.Scatter3d(
        x=p_x,
        y=p_y,
        z=p_z,
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            colorscale='Viridis',
            opacity=0.8
        ),
        name=name
    )


def plot_3d_graph(graphs: np.array, with_markers: bool = False, colors: list = [], names: list = []):
    """
    Plotting 3d graph
    """

    seaborn.set_style("darkgrid")

    traces = list()

    for i in range(len(graphs)):
        points = graphs[i]
        p_x = [point[0] for point in points]
        p_y = [point[1] for point in points]
        p_z = [point[2] for point in points]

        color = 'black' if len(colors) < i else colors[i]
        name = '' if len(names) < i else names[i]

        if with_markers:
            traces.append(_add_markers_3d(points, 3, color=color, name=name))

        traces.append(
            go.Scatter3d(
                x=p_x,
                y=p_y,
                z=p_z,
                mode='lines',
                line=dict(color=color, width=1),
                name=name
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        autosize=False,
        width=900,
        height=900
    )
    fig.show()
