from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_geometric.data import Data


def training_loop(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  data: Data, epochs: int, patience: int) -> torch.nn.Module:
    """
    Training loop for the neural network model.
    :param model: neural network model to be trained.
    :param optimizer: optimizer for training (Adam is used in this work).
    :param data: data for training.
    :param epochs: number of epochs.
    :param patience: patience parameter for early stopping during training; defaults to 10.
    :return: trained model.
    """
    max_acc = 0
    best_state_dict = None
    streak = 0
    for epoch in range(epochs):
        train(model, optimizer, data)
        acc = evaluate(model, data, test=False)['val_acc']
        if acc >= max_acc:
            max_acc = acc
            best_state_dict = deepcopy(model.state_dict())
            streak = 0
        else:
            streak += 1
        if streak >= patience:
            model.load_state_dict(best_state_dict)
            return model
    model.load_state_dict(best_state_dict)
    return model


def train(model: torch.nn.Module, optimizer: Optimizer, data: Data) -> float:
    """
    Perform one iteration of training the model.
    :param model: neural network model to be trained.
    :param optimizer: optimizer for training (Adam is used in this work).
    :param data: data for training.
    :return: training loss.
    """
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model: torch.nn.Module, data: Data, test: bool) -> dict:
    """
    Evaluate the model.
    :param model: neural network model to be evaluated.
    :param data: data for evaluation.
    :param test: whether to evaluate on training data; if False, evaluation is done on validation data.
    :return: dictionary containing validation/test accuracy.
    """
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
