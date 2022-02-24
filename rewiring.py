import argparse
import pickle

from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from sdrf import sdrf_w_cuda
from utils import load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CGNN')
    parser.add_argument('--data_path', type=str, help="Path to the data files.", default='./data')
    parser.add_argument('--dataset', type=str, help="Name of the dataset", required=True)
    args = parser.parse_args()

    dt = load_data(args.data_path, args.dataset)
    G = sdrf_w_cuda(dt, is_undirected=True, tau=0.7, loops=200)

    with open('rewiring_edges', 'wb') as f:
        pickle.dump(G.edge_index, f)
