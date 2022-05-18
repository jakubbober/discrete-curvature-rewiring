import pickle

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_geometric.data import Data

from bronstein_paper.gcn import GCN
from bronstein_paper.optimizer import get_optimizer
from bronstein_paper.sdrf import SDRFDataset
from bronstein_paper.seeds import val_seeds, test_seeds
from bronstein_paper.splits import set_train_val_test_split


def training_loop(model, optimizer, data, epochs, patience):
    max_acc = 0
    best_state_dict = None
    streak = 0
    for epoch in range(epochs):
        train(model, optimizer, data)
        acc = evaluate(model, data, test=False)['val_acc']
        if acc >= max_acc:
            best_epoch = epoch
            max_acc = acc
            best_state_dict = model.state_dict()
            streak = 0
        else:
            streak += 1
        if streak >= patience:
            model.load_state_dict(best_state_dict)
            return model
    print(best_epoch)
    return model


def train(model: torch.nn.Module, optimizer: Optimizer, data: Data):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model: torch.nn.Module, data: Data, test: bool):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    eval_dict = {}
    keys = ['val', 'test'] if test else ['val']
    for key in keys:
        mask = data[f'{key}_mask']
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        eval_dict[f'{key}_acc'] = acc
    return eval_dict


if __name__ == '__main__':
    max_steps_Cora_sdrf = 100
    tau_cora_sdrf = 163
    removal_bound_Cora_sdrf = 0.95
    dropout_Cora_sdrf = 0.3396
    hidden_Cora_sdrf = [128]
    lr_Cora_sdrf = 0.0244
    decay_Cora_sdrf = 0.1076
    dataset = SDRFDataset(max_steps=max_steps_Cora_sdrf, data_dir='dt', undirected=True, remove_edges=True,
                          tau=tau_cora_sdrf, removal_bound=removal_bound_Cora_sdrf)
    eval_dicts = []
    for seed in val_seeds:
        model = GCN(dataset, dropout=dropout_Cora_sdrf, hidden=hidden_Cora_sdrf)
        optimizer = get_optimizer('adam', model, lr_Cora_sdrf, decay_Cora_sdrf)
        data = set_train_val_test_split(seed, dataset.data)
        dataset.data = data
        model = training_loop(model, optimizer, data, 1000)
        eval_dicts.append(model.state_dict())
    with open('eval_dicts/eval_dicts_.pk', 'wb') as f:
        pickle.dump(eval_dicts, f)
