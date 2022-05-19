import os
import pickle

from bronstein_paper.base import BaseDataset
from bronstein_paper.cuda import sdrf
from bronstein_paper.gcn import GCN
from bronstein_paper.node_classification import training_loop
from bronstein_paper.optimizer import get_optimizer
from bronstein_paper.sdrf_no_cuda import sdrf_no_cuda
from bronstein_paper.seeds import val_seeds
from bronstein_paper.splits import set_train_val_test_split, set_train_val_test_split_frac
from curvature.frc_rewritten import compute_ricci_curvature, compute_ricci_curvature_edge

removal_bound_Cora_sdrf = 0.95


def evaluate_curvatures(dname, curv_type, removal_bound, redo_rewiring=False):
    max_steps_Cora_sdrf = 100
    tau_cora_sdrf = 163
    dropout_Cora_sdrf = 0.3396
    hidden_Cora_sdrf = [128]
    lr_Cora_sdrf = 0.0244
    decay_Cora_sdrf = 0.1076

    dataset = BaseDataset(dname, undirected=True, data_dir='dt')

    def rewire(dt):
        if curv_type == 'bfc':
            dt = sdrf(dt, loops=max_steps_Cora_sdrf, removal_bound=removal_bound, tau=tau_cora_sdrf,
                        is_undirected=True)
        elif curv_type is None:
            dt = sdrf_no_cuda(dt, curv_type, loops=0, removal_bound=removal_bound, tau=tau_cora_sdrf)
        else:
            dt = sdrf_no_cuda(dt, curv_type, loops=max_steps_Cora_sdrf, removal_bound=removal_bound,
                                tau=tau_cora_sdrf)
        return dt.edge_index

    if not redo_rewiring:
        dataset.data.edge_index = rewire(dataset.data)

        os.makedirs(f'edge_indexes/{dname}', exist_ok=True)
        with open(f'edge_indexes/{dname}/edge_index_{curv_type}.pk', 'wb') as f:
            pickle.dump(dataset.data.edge_index, f)

    state_dicts = []

    for i, seed in enumerate(val_seeds):
        if redo_rewiring:
            dataset.data.edge_index = rewire(dataset.data)
            os.makedirs(f'edge_indexes/{dname}_redo_rewiring/{curv_type}', exist_ok=True)
            with open(f'edge_indexes/{dname}_redo_rewiring/{curv_type}/edge_index_{curv_type}_{str(0)+str(i) if i < 10 else str(i)}.pk', 'wb') as f:
                pickle.dump(dataset.data.edge_index, f)
        if dname in ('Cora', 'Citeseer', 'Pubmed'):
            data = set_train_val_test_split(seed, dataset.data)
        else:
            data = set_train_val_test_split_frac(seed, dataset.data, val_frac=0.2, test_frac=0.2)
        dataset.data = data

        model = GCN(dataset, hidden_Cora_sdrf, dropout_Cora_sdrf)
        optimizer = get_optimizer('adam', model, lr_Cora_sdrf, decay_Cora_sdrf)

        model = training_loop(model, optimizer, dataset.data, epochs=1000, patience=100)
        state_dicts.append(model.state_dict())

    return state_dicts


if __name__ == '__main__':
    for name in ['Pubmed', 'CS', 'Computers', 'Photo', 'CoauthorCS', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor']:
        for curvature, rb in [(None, None), ('1d', -1), ('augmented', -1), ('haantjes', -1),
                              ('bfc', removal_bound_Cora_sdrf)]:
            sd = evaluate_curvatures(name, curvature, rb)
            os.makedirs(f'state_dicts/{name}', exist_ok=True)
            with open(f'state_dicts/{name}/state_dicts_{curvature}.pk', 'wb') as f:
                pickle.dump(sd, f)
