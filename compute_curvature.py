import argparse
from time import time

from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from torch_geometric.utils import to_networkx

from utils import load_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CGNN')
    parser.add_argument('--data_path', type=str, help="Path to the data files.", default='./data')
    parser.add_argument('--dataset', type=str, help="Name of the dataset", required=True)
    args = parser.parse_args()

    dt = load_data(args.data_path, args.dataset)

    G = to_networkx(dt, to_undirected=dt.is_undirected())
    orc = OllivierRicci(G, alpha=0.5)
    frc = FormanRicci(G, method="1d")
    start = time()
    orc.compute_ricci_curvature()
    print("Ollivier Ricci time:", time() - start)
    start = time()
    frc.compute_ricci_curvature()
    print("Forman Ricci time:", time() - start)
    out_ollivier = ""
    out_forman = ""
    for v1, v2 in orc.G.edges:
        out_ollivier += f"{v1} {v2} {orc.G[v1][v2]['ricciCurvature']}\n"
    for v1, v2 in frc.G.edges:
        out_forman += f"{v1} {v2} {frc.G[v1][v2]['formanCurvature']}\n"
    with open("graph_olliver.edge_list", "w") as f:
        f.write(out_ollivier)
    with open("graph_forman.edge_list", "w") as f:
        f.write(out_forman)
