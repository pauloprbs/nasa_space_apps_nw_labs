"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import sys


class Optimizer:
    def __init__(self,in_features,BATCHSIZE,CLASSES,DIR,EPOCHS,N_TRAIN_EXAMPLES,N_VALID_EXAMPLES,predict_event):
        DEVICE = torch.device("cpu")
        self.BATCHSIZE = BATCHSIZE
        self.CLASSES = CLASSES
        self.DIR = DIR
        self.EPOCHS = EPOCHS
        self.N_TRAIN_EXAMPLES = N_TRAIN_EXAMPLES
        self.N_VALID_EXAMPLES = N_VALID_EXAMPLES
        self.predict_event = predict_event
        self.in_features = in_features

    def setup(self, model):
        if self.name == "Adam":
            return optim.Adam(model.parameters(), lr=self.lr)
        elif self.name == "RMSprop":
            return optim.RMSprop(model.parameters(), lr=self.lr)
        else:
            return optim.SGD(model.parameters(), lr=self.lr)

    def get_data(self):
        pass

    def create_Linear(self,trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []

        out_features = trial.suggest_int("n_units_l0", self.in_features//4, self.in_features//2)
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(self.in_features, out_features))
                layers.append(nn.ReLU())
                p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
                layers.append(nn.Dropout(p))

            else:
                layers.append(nn.Linear(out_features//i, out_features//i+1))
                layers.append(nn.ReLU())
                p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
                layers.append(nn.Dropout(p))

            # self.in_features = out_features
        layers.append(nn.Linear(out_features//(n_layers+1), self.CLASSES))
        # layers.append(nn.LogSoftmax(dim=1))

        return nn.Sequential(*layers)

    def create_Conv(self,trial):
        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        n_convs = trial.suggest_int("n_convolutions", 1, 3)
        
        layers = []
        in_channels = self.in_features
        out_features = trial.suggest_int("out_features", in_channels//8,in_channels//2)
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        stride = trial.suggest_int("stride", 1, 3)
        padding = trial.suggest_int("padding", 1, 3)
        predict_event = self.predict_event
        for i in range(n_convs):
            if i == 0:
                layers.append(nn.Conv1d(in_channels, out_features, kernel_size=kernel_size, stride=stride, padding=padding))
                layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding))
                layers.append(nn.SELU())
                p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
                layers.append(nn.Dropout(p))
            else:
                layers.append(nn.Conv1d(out_features//i, out_features//i+1, kernel_size=kernel_size, stride=stride, padding=padding))
                layers.append(nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding))
                layers.append(nn.SELU())
                p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
                layers.append(nn.Dropout(p))

        layers.append(nn.Linear(out_features//(n_convs+1), self.CLASSES))
        if predict_event:
            layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)
    
    def objective(self,trial):
        # Generate the model.
        model = self.define_model(trial).to(self.DEVICE)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        train_loader, valid_loader = self.get_data()

        # Training of the model.
        for epoch in range(self.EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # Limiting training data for faster epochs.
                if batch_idx * self.BATCHSIZE >= self.N_TRAIN_EXAMPLES:
                    break

                data, target = data.view(data.size(0), -1).to(self.DEVICE), target.to(self.DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

            # Validation of the model.
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    # Limiting validation data.
                    if batch_idx * self.BATCHSIZE >= self.N_VALID_EXAMPLES:
                        break
                    data, target = data.view(data.size(0), -1).to(self.DEVICE), target.to(self.DEVICE)
                    output = model(data)
                    # Get the index of the max log-probability.
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy = correct / min(len(valid_loader.dataset), self.N_VALID_EXAMPLES)

            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return accuracy
