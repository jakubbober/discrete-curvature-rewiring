import pickle
import random
from typing import Tuple, List

import numpy as np
import pandas as pd

from experiment.data_loader import DataLoader
from experiment.data_splits import set_train_val_test_split, set_train_val_test_split_frac
from experiment.training_loop import evaluate
from models.gcn import GCN
from utils.hyperparams import hyperparams
from utils.seeds import val_seeds, test_seeds


def test_performance(dname: str, curv_type: str, test: bool = True,
                     redo_rewiring: bool = False) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """
    Test the performance of saved models (from experiment.save_models.py) on the val/test set for val/test seeds.
    :param dname: name of the dataset.
    :param curv_type: type of discrete curvature.
    :param test: whether to test performance on the test set and for test seeds;
                 if False, evaluates on the eval set and for eval seeds; defaults to True.
    :param redo_rewiring: whether the data to be tested was saved with the flag redo_rewiring set to True
                          when saving models with save_models from experiment.save_models.py; defaults to False
    :return: [1] list test/val accuracies for each seed.
             [2] mean of all test/val accuracies.
             [3] standard deviation of all test/val accuracies.
    """
    # Fetch corresponding hyperparameters.
    dropout = hyperparams[dname]['dropout']
    hidden_depth = hyperparams[dname]['hidden_depth']
    hidden_dim = hyperparams[dname]['hidden_dim']

    # Fetch corresponding state dictionary.
    if curv_type is not None:
        with open(f'state_dicts/{dname}/state_dicts_{str(curv_type)}_best.pk', 'rb') as f:
            state_dict = pickle.load(f)
    else:
        with open(f'state_dicts/{dname}/state_dicts_{str(curv_type)}.pk', 'rb') as f:
            state_dict = pickle.load(f)

    dataset = DataLoader(dname, undirected=True, data_dir='dt')

    # If redo_rewiring is set to False, fetch corresponding graph edge index.
    if curv_type is not None:
        with open(f'edge_indices/{dname}/edge_index_{str(curv_type)}_best.pk', 'rb') as f:
            edge_index = pickle.load(f)
            dataset.data.edge_index = edge_index
    else:
        with open(f'edge_indices/{dname}/edge_index_{str(curv_type)}.pk', 'rb') as f:
            edge_index = pickle.load(f)
            dataset.data.edge_index = edge_index

    val_accs = []
    test_accs = []
    
    best_val_acc = 0
    seed = random.randint(1, 10000)
    # i = -1
    # for seed, state_dict in zip(val_seeds, state_dicts) if not test else zip(test_seeds, state_dicts):
    #     random.seed(seed)
    #     i += 1
    # If redo_rewiring is set to True, fetch corresponding graph edge index for current iteration.
    # if redo_rewiring and curv_type is not None:
    #     with open(f'edge_indices/{dname}_redo_rewiring/{str(curv_type)}/edge_index_{str(curv_type)}'
    #               f'_{str(0) + str(i) if i < 10 else str(i)}.pk', 'rb') as f:
    #         edge_index = pickle.load(f)
    #         dataset.data.edge_index = edge_index
    if dname in ('Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CoauthorCS'):
        data = set_train_val_test_split(seed, dataset.data)
    else:
        data = set_train_val_test_split_frac(seed, dataset.data, val_frac=0.2, test_frac=0.2)
    dataset.data = data

    # Fetch corresponding state dictionary and evaluate the model
    model = GCN(dataset=dataset, hidden=[hidden_dim] * hidden_depth, dropout=dropout)
    model.load_state_dict(state_dict)

    ed = evaluate(model, dataset.data, test=test)

    if redo_rewiring and curv_type is not None and ed['val_acc'] > best_val_acc:
        with open(f'edge_indices/{dname}/edge_index_{str(curv_type)}_best.pk', 'wb') as f:
            pickle.dump(edge_index, f)
        with open(f'state_dicts/{dname}/state_dicts_{str(curv_type)}_best.pk', 'wb') as f:
            pickle.dump(state_dict, f)
        best_val_acc = ed['val_acc']

    if not test:
        val_accs.append(ed['val_acc'])
    else:
        test_accs.append(ed['test_acc'])

    if not test:
        return val_accs, np.mean(val_accs), np.std(val_accs)
    else:
        return test_accs, np.mean(test_accs), np.std(test_accs)


if __name__ == '__main__':
    datasets = ['Cora', 'Citeseer', 'Cornell', 'Texas', 'Wisconsin']
    curvatures = [None, '1d', 'augmented', 'haantjes', 'bfc']
    result = {d: [] for d in datasets}
    for name in datasets:
        print(name)
        for curvature in curvatures:
            print(str(curvature) + ":")
            # try:
            out = test_performance(name, curvature, test=True, redo_rewiring=False)
            cell = str(round(out[1] * 100, 2)) + " +- " + str(round(out[2] * 100 * 0.196, 2))
            # except:
                # If a corresponding file is not found, record error value.
                # For example, there may have been a CUDA OOM error when calculating BFC for a dataset.
                # cell = 'ERROR'
            result[name].append(cell)
            print(cell)
            print()
        print()
    pd.DataFrame(result, index=[str(c) for c in curvatures]).to_excel('results.xlsx')
