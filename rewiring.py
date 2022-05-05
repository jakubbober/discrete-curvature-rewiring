import argparse
import pickle

from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from sdrf_rewritten import sdrf
from utils.load_data import load_data

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='CGNN')
    # parser.add_argument('--data_path', type=str, help="Path to the data files.", default='./data')
    # parser.add_argument('--dataset', type=str, help="Name of the dataset", required=True)
    # args = parser.parse_args()

    # dt = load_data(args.data_path, args.dataset)
    dt = load_data('data/', 'Cora')
    G = sdrf(dt, curv_type='bfc', max_iter=200, C=-50)

    with open('rewiring_edges', 'wb') as f:
        pickle.dump(G.edge_index, f)
