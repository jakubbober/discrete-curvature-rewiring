import os
import pickle
import time

from tqdm import tqdm
from bronstein_paper.base import BaseDataset
from bronstein_paper.cuda import sdrf
from bronstein_paper.gcn import GCN
from bronstein_paper.hyperparams import hyperparams
from bronstein_paper.node_classification import training_loop
from bronstein_paper.optimizer import get_optimizer
from bronstein_paper.sdrf_no_cuda import sdrf_no_cuda
from bronstein_paper.seeds import val_seeds
from bronstein_paper.splits import set_train_val_test_split, set_train_val_test_split_frac
from curvature.frc_rewritten import compute_ricci_curvature, compute_ricci_curvature_edge


def evaluate_curvatures(dname, curv_type, redo_rewiring=False):
    dropout = hyperparams[dname]['dropout']
    hidden_depth = hyperparams[dname]['hidden_depth']
    hidden_dim = hyperparams[dname]['hidden_dim']
    learning_rate = hyperparams[dname]['learning_rate']
    weight_decay = hyperparams[dname]['weight_decay']
    max_iterations = hyperparams[dname]['max_iterations']
    tau = hyperparams[dname]['tau']
    removal_bound = hyperparams[dname]['removal_bound']


    dataset = BaseDataset(dname, undirected=True, data_dir='dt')

    def rewire(dt):
        start = time.time()
        if curv_type == 'bfc':
            dt = sdrf(dt, loops=max_iterations, removal_bound=removal_bound, tau=tau, is_undirected=True)
        elif curv_type is not None:
            dt = sdrf_no_cuda(dt, curv_type, loops=max_iterations, removal_bound=removal_bound, tau=tau)
        if curv_type is not None:
            print(curv_type, dname, 'time:', str(time.time() - start))
        return dt.edge_index

    if not redo_rewiring:
        dataset.data.edge_index = rewire(dataset.data)

        os.makedirs(f'edge_indexes_hyperparams/{dname}', exist_ok=True)
        with open(f'edge_indexes_hyperparams/{dname}/edge_index_{curv_type}.pk', 'wb') as f:
            pickle.dump(dataset.data.edge_index, f)

    state_dicts = []

    for i, seed in enumerate(tqdm(val_seeds)):
        if redo_rewiring:
            dataset.data.edge_index = rewire(dataset.data)
            os.makedirs(f'edge_indexes_hyperparams/{dname}_redo_rewiring/{curv_type}', exist_ok=True)
            with open(f'edge_indexes_hyperparams/{dname}_redo_rewiring/{curv_type}/edge_index_{curv_type}_{str(0)+str(i) if i < 10 else str(i)}.pk', 'wb') as f:
                pickle.dump(dataset.data.edge_index, f)
        if dname in ('Cora', 'Citeseer', 'Pubmed'):
            data = set_train_val_test_split(seed, dataset.data)
        else:
            data = set_train_val_test_split_frac(seed, dataset.data, val_frac=0.2, test_frac=0.2)
        dataset.data = data

        model = GCN(dataset=dataset, hidden=[hidden_dim] * hidden_depth, dropout=dropout)
        optimizer = get_optimizer('adam', model, learning_rate, weight_decay)

        model = training_loop(model, optimizer, dataset.data, epochs=1000, patience=10)
        state_dicts.append(model.state_dict())

    return state_dicts


if __name__ == '__main__':
    for name in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CoauthorCS', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor']:
        for curvature in (None, '1d', 'augmented', 'haantjes', 'bfc'):
            try:
                sd = evaluate_curvatures(name, curvature)
                os.makedirs(f'state_dicts_hyperparams/{name}', exist_ok=True)
                with open(f'state_dicts_hyperparams/{name}/state_dicts_{curvature}.pk', 'wb') as f:
                    pickle.dump(sd, f)
            except Exception as e:
                print(str(name), str(curvature), str(e))
