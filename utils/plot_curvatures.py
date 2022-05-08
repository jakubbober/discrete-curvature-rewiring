import matplotlib

import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('WebAgg')


def plot_curvatures(curv_type: str, d_name: str, normalize: bool = False) -> None:
    """
    Plot given type of curvature for given input data.
    :param curv_type: type of curvature ('Ollivier', 'Forman' or 'BalancedForman').
    :param d_name: dataset name.
    :param normalize: if True, normalize the curvature points to have zero mean and unit variance.
    """
    with open(f'../data/{curv_type}/graph_{d_name}.edge_list', 'r') as f:
        curvatures = np.array([float(line.split(' ')[-1].replace("\n", "")) for line in f])
        if normalize:
            curvatures = (curvatures - curvatures.mean()) / curvatures.std()
        plt.scatter(range(len(curvatures)), curvatures)
        plt.title(f'{curv_type} Curvature')


if __name__ == '__main__':
    for i, curvature in enumerate(('Ollivier', 'Forman', 'BalancedForman')):
        plt.figure(i)
        plot_curvatures(curvature, 'Cora')
    plt.show()
