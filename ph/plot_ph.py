import pickle

import matplotlib
import numpy as np
import plotly.graph_objects
from gtda.plotting import plot_diagram

matplotlib.use('WebAgg')


def plot_ph(ph: np.ndarray) -> plotly.graph_objects.Figure:
    """
    Plot the 1D persistence homology.
    :param ph: numpy array of 1D persistence homology points.
    :return: figure representing the persistence diagram.
    """
    ph_w_dim = np.hstack((ph, np.ones([len(ph), 1])))
    fig = plot_diagram(ph_w_dim, homology_dimensions=tuple([1]))
    return fig


if __name__ == '__main__':
    with open('ph_before', 'rb') as f:
        ph_before = pickle.load(f)
    with open('ph_after', 'rb') as f:
        ph_after = pickle.load(f)

    fig_before = plot_ph(ph_before)
    fig_after = plot_ph(ph_after)

    fig_before.show()
    fig_after.show()
