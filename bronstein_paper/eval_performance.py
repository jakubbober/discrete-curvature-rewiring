import pickle

import numpy as np

from bronstein_paper.base import BaseDataset
from bronstein_paper.gcn import GCN
from bronstein_paper.node_classification import evaluate
from bronstein_paper.seeds import val_seeds, test_seeds
from bronstein_paper.splits import set_train_val_test_split, set_train_val_test_split_frac


def eval_performance(dname, curv_type, test=False, redo_rewiring=False):
    max_steps_Cora_sdrf = 100
    tau_cora_sdrf = 163
    removal_bound_Cora_sdrf = 0.95
    dropout_Cora_sdrf = 0.3396
    hidden_Cora_sdrf = [128]
    lr_Cora_sdrf = 0.0244
    decay_Cora_sdrf = 0.1076

    with open(f'state_dicts/{dname}/state_dicts_{curv_type}.pk', 'rb') as f:
        state_dicts = pickle.load(f)

    dataset = BaseDataset(dname, undirected=True, data_dir='dt')

    if not redo_rewiring:
        with open(f'edge_indexes/{dname}/edge_index_{curv_type}.pk', 'rb') as f:
            edge_index = pickle.load(f)
            dataset.data.edge_index = edge_index

    val_accs = []
    test_accs = []

    i = -1
    for seed, state_dict in zip(val_seeds, state_dicts) if not test else zip(test_seeds, state_dicts):
        i += 1
        if redo_rewiring:
            with open(f'edge_indexes/{dname}_redo_rewiring/{curv_type}/edge_index_{curv_type}_{str(0)+str(i) if i < 10 else str(i)}.pk', 'rb') as f:
                edge_index = pickle.load(f)
                dataset.data.edge_index = edge_index
        if dname in ('Cora', 'Citeseer', 'Pubmed'):
            data = set_train_val_test_split(seed, dataset.data)
        else:
            data = set_train_val_test_split_frac(seed, dataset.data, val_frac=0.2, test_frac=0.2)
        dataset.data = data

        model = GCN(dataset, hidden_Cora_sdrf, dropout_Cora_sdrf)
        model.load_state_dict(state_dict)

        ed = evaluate(model, dataset.data, test=test)

        if not test:
            val_accs.append(ed['val_acc'])
        else:
            test_accs.append(ed['test_acc'])

    if not test:
        return val_accs, np.mean(val_accs), np.std(val_accs)
    else:
        return test_accs, np.mean(test_accs), np.std(test_accs)


if __name__ == '__main__':
    for name in ['Cora', 'Citeseer', 'Pubmed', 'CS', 'Computers', 'Photo', 'CoauthorCS', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor']:
        print(name)
        for curvature in [None, '1d', 'augmented', 'haantjes', 'bfc']:
            print(str(curvature) + ":")
            for out in eval_performance(name, curvature, test=True):
                print(out)
            print()
        print()
