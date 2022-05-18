import pickle

from bronstein_paper.base import BaseDataset
from bronstein_paper.cuda import sdrf
from bronstein_paper.gcn import GCN
from bronstein_paper.node_classification import training_loop
from bronstein_paper.optimizer import get_optimizer
from bronstein_paper.sdrf_no_cuda import sdrf_no_cuda
from bronstein_paper.seeds import val_seeds
from bronstein_paper.splits import set_train_val_test_split
from curvature.frc_rewritten import compute_ricci_curvature, compute_ricci_curvature_edge

def evaluate_curvatures(curv_type):
    max_steps_Cora_sdrf = 100
    tau_cora_sdrf = 163
    removal_bound_Cora_sdrf = 0.95
    dropout_Cora_sdrf = 0.3396
    hidden_Cora_sdrf = [128]
    lr_Cora_sdrf = 0.0244
    decay_Cora_sdrf = 0.1076

    dataset = BaseDataset(undirected=True, data_dir='dt')
    data = dataset.data

    if curv_type == 'bfc':
        data = sdrf(data, loops=max_steps_Cora_sdrf, removal_bound=removal_bound_Cora_sdrf, tau=tau_cora_sdrf, is_undirected=True)
    else:
        data = sdrf_no_cuda(data, curv_type, loops=max_steps_Cora_sdrf, removal_bound=removal_bound_Cora_sdrf, tau=tau_cora_sdrf)

    dataset.data = data

    eval_dicts = []

    for seed in val_seeds:
        data = set_train_val_test_split(seed, dataset.data)
        dataset.data = data

        model = GCN(dataset, hidden_Cora_sdrf, dropout_Cora_sdrf)
        optimizer = get_optimizer('adam', model, lr_Cora_sdrf, decay_Cora_sdrf)

        model = training_loop(model, optimizer, data, epochs=1000, patience=100)
        eval_dicts.append(model.state_dict())

    return eval_dicts


if __name__ == '__main__':
    for curvature in ['bfc', '1d', 'augmented', 'haantjes']:
        eval_dicts = evaluate_curvatures(curvature)
        with open(f'eval_dicts/eval_dicts_{curvature}.pk', 'wb') as f:
            pickle.dump(eval_dicts, f)
