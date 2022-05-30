import os
import pickle
import time
from typing import List

import torch.optim
from tqdm import tqdm

from experiment.data_loader import DataLoader
from experiment.data_splits import set_train_val_test_split, set_train_val_test_split_frac
from experiment.training_loop import training_loop
from models.gcn import GCN
from rewiring.sdrf_cuda_bfc import sdrf_cuda_bfc
from rewiring.sdrf_no_cuda import sdrf_no_cuda
from utils.hyperparams import hyperparams
from utils.seeds import val_seeds


def save_models(dname: str, curv_type: str, patience: int = 10, redo_rewiring: bool = False) -> List[dict]:
    """
    Find the best models for all curvature types and seeds for a dataset together with the rewired edges indices.
    :param dname: name of the dataset.
    :param curv_type: type of discrete curvature.
    :param patience: patience parameter for early stopping during training; defaults to 10.
    :param redo_rewiring: whether to redo the rewiring for each seed; defaults to False.
    :return: list of model state dictionaries for all seeds.
    """
    # Fetch corresponding hyperparameters.
    dropout = hyperparams[dname]['dropout']
    hidden_depth = hyperparams[dname]['hidden_depth']
    hidden_dim = hyperparams[dname]['hidden_dim']
    learning_rate = hyperparams[dname]['learning_rate']
    weight_decay = hyperparams[dname]['weight_decay']
    max_iterations = hyperparams[dname]['max_iterations']
    tau = hyperparams[dname]['tau']
    removal_bound = hyperparams[dname]['removal_bound']

    # Create an undirected dataset.
    dataset = DataLoader(dname, undirected=True, data_dir='dt')

    # Perform graph rewiring, print out rewiring time.
    def rewire(dt):
        start = time.time()
        if curv_type == 'bfc':
            dt = sdrf_cuda_bfc(dt, loops=max_iterations, remove_edges=True,
                               removal_bound=removal_bound, tau=tau, is_undirected=True)
        elif curv_type is not None:
            dt = sdrf_no_cuda(dt, curv_type, loops=max_iterations, remove_edges=True,
                              removal_bound=removal_bound, tau=tau)
        if curv_type is not None:
            print(curv_type, dname, 'time:', str(time.time() - start))
        return dt.edge_index

    # If redo_rewiring is set to False, save graph edge index.
    if not redo_rewiring:
        dataset.data.edge_index = rewire(dataset.data)

        os.makedirs(f'edge_indices/{dname}', exist_ok=True)
        with open(f'edge_indices/{dname}/edge_index_{curv_type}.pk', 'wb') as f:
            pickle.dump(dataset.data.edge_index, f)

    state_dicts = []

    # Save the models performing best on validation set using early stopping.
    for i, seed in enumerate(tqdm(val_seeds)):
        # If redo_rewiring is set to True, save the edge indices for current iteration.
        if redo_rewiring:
            dataset.data.edge_index = rewire(dataset.data)
            os.makedirs(f'edge_indices/{dname}_redo_rewiring/{curv_type}', exist_ok=True)
            with open(f'edge_indices/{dname}_redo_rewiring/{curv_type}/edge_index_{curv_type}'
                      f'_{str(0) + str(i) if i < 10 else str(i)}.pk', 'wb') as f:
                pickle.dump(dataset.data.edge_index, f)

        # Split the data accordingly.
        if dname in ('Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CoauthorCS'):
            # TODO remove duplication with test_performance.py
            data = set_train_val_test_split(seed, dataset.data)
        else:
            data = set_train_val_test_split_frac(seed, dataset.data, val_frac=0.2, test_frac=0.2)
        dataset.data = data

        model = GCN(dataset=dataset, hidden=[hidden_dim] * hidden_depth, dropout=dropout)

        # Reproduce optimizer params like in
        # https://github.com/jctops/understanding-oversquashing/blob/main/gdl/src/gdl/experiment/optimizer.py.
        optimizer_params = [
            {'params': model.non_reg_params, 'weight_decay': 0},
            {'params': model.reg_params, 'weight_decay': weight_decay}
        ]
        optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)

        # Train the model and record state dictionaries.
        model = training_loop(model, optimizer, dataset.data, epochs=1000, patience=patience)
        state_dicts.append(model.state_dict())

    return state_dicts


if __name__ == '__main__':
    datasets = ['Computers', 'Photo', 'CoauthorCS']
    curvatures = [None, '1d', 'augmented', 'haantjes', 'bfc']

    for name in datasets:
        for curvature in curvatures:
            try:
                sd = save_models(name, curvature)
                os.makedirs(f'state_dicts_hyperparams/{name}', exist_ok=True)
                with open(f'state_dicts_hyperparams/{name}/state_dicts_{curvature}.pk', 'wb') as file:
                    pickle.dump(sd, file)
            except Exception as e:
                print(str(name), str(curvature), str(e))
