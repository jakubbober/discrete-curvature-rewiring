import matplotlib
import tornado
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_curvatures(ollivier, forman, balanced):
    # plt.interactive(False)
    with open(ollivier, 'r') as ol, open(forman, 'r') as fr, open(balanced, 'r') as bl:
        for i, file in enumerate([ol, fr, bl]):
            curvatures = np.array([float(line.split(' ')[-1].replace("\n", "")) for line in file])
            # curvatures = (curvatures - curvatures.mean()) / curvatures.std()
            plt.figure(i)
            plt.scatter(range(len(curvatures)), curvatures)
            # plt.legend(*scatter.legend_elements())
    plt.show()


if __name__ == '__main__':
    plot_curvatures('data/Ollivier/graph_Cora.edge_list', 'data/Forman/graph_Cora.edge_list', 'data/BalancedForman/graph_Cora.edge_list')
