from copy import deepcopy

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class FullyConnectedNeuralNetwork:
    """
    Fully connected neural network for multi-output regression.
    """

    def __init__(
        self,
        hidden_layers=(64,),
        activation="relu",
        learning_rate=0.001,
        epochs=400,
        batch_size=1024,
        weight_decay=0.0,
        shuffle=True,
        valid_frac=0.1,
        random_state=None,
        verbose=False,
    ):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.shuffle = shuffle
        self.valid_frac = valid_frac
        self.random_state = random_state
        self.verbose = verbose

        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scaler_x = None
        self.scaler_y = None

        self.history_ = None
        self.best_epoch_ = None
        self.best_loss_valid_ = None

    def fit(self, x, y):
        n_features = x.shape[1]
        n_targets = y.shape[1]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        x_train, x_valid, y_train, y_valid = train_test_split(
            x,
            y,
            test_size=self.valid_frac,
            random_state=self.random_state,
        )

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.scaler_x.fit(x_train)
        self.scaler_y.fit(y_train)

        x_train = self.scaler_x.transform(x_train)
        y_train = self.scaler_y.transform(y_train)

        x_valid = self.scaler_x.transform(x_valid)
        y_valid = self.scaler_y.transform(y_valid)

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        x_valid = torch.tensor(x_valid, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)

        dataset_train = TensorDataset(
            x_train,
            y_train,
        )

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        self.build_model(
            n_features,
            n_targets,
        )

        best_loss_valid = None
        best_model_state = None
        best_epoch = None

        epochs_record = []
        loss_train_record = []
        loss_valid_record = []

        for epoch in range(
            1,
            self.epochs + 1,
        ):
            loss_train = self.train_iteration(
                dataloader_train
            )

            loss_valid = self.valid_iteration(
                x_valid,
                y_valid,
            )

            epochs_record.append(
                epoch
            )

            loss_train_record.append(
                loss_train
            )

            loss_valid_record.append(
                loss_valid
            )

            if (
                best_loss_valid is None
                or loss_valid < best_loss_valid
            ):
                best_loss_valid = loss_valid
                best_model_state = deepcopy(
                    self.model.state_dict()
                )
                best_epoch = epoch

            if self.verbose:
                print(
                    f"Epoch: {epoch:5d} | "
                    f"Loss (Train): {loss_train:10.5f} | "
                    f"Loss (Valid): {loss_valid:10.5f}"
                )

        self.build_model(
            n_features,
            n_targets,
        )

        self.model.load_state_dict(
            best_model_state
        )

        self.history_ = {
            "epoch": epochs_record,
            "loss_train": loss_train_record,
            "loss_valid": loss_valid_record,
        }

        self.best_epoch_ = best_epoch
        self.best_loss_valid_ = best_loss_valid

        return self

    def predict(self, x):
        x = self.scaler_x.transform(
            x
        )

        x = torch.tensor(
            x,
            dtype=torch.float32,
        )

        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(
                x
            )

        y_pred = (
            y_pred.detach()
            .cpu()
            .numpy()
        )

        y_pred = (
            self.scaler_y
            .inverse_transform(
                y_pred
            )
        )

        return y_pred

    def build_model(
        self,
        n_features,
        n_targets,
    ):
        valid_activations = {
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
        }

        if self.activation not in valid_activations:
            raise ValueError(
                f"Unknown activation '{self.activation}'."
            )

        nn_activation = (
            valid_activations[
                self.activation
            ]
        )

        seq_stack = []

        n_nodes_in = n_features

        for n_nodes_out in self.hidden_layers:
            seq_stack.append(
                nn.Linear(
                    n_nodes_in,
                    n_nodes_out,
                )
            )

            seq_stack.append(
                nn_activation()
            )

            n_nodes_in = n_nodes_out

        seq_stack.append(
            nn.Linear(
                n_nodes_in,
                n_targets,
            )
        )

        self.model = nn.Sequential(
            *seq_stack
        )

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def train_iteration(
        self,
        dataloader_train,
    ):
        loss_sum = 0.0
        n_samples = 0

        self.model.train()

        for (
            x_batch_train,
            y_batch_train,
        ) in dataloader_train:

            y_batch_pred = self.model(
                x_batch_train
            )

            loss_batch = self.loss_fn(
                y_batch_pred,
                y_batch_train,
            )

            loss_batch.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_size = (
                x_batch_train.shape[0]
            )

            loss_sum += (
                loss_batch.item()
                * batch_size
            )

            n_samples += batch_size

        return (
            loss_sum
            / n_samples
        )

    def valid_iteration(
        self,
        x_valid,
        y_valid,
    ):
        self.model.eval()

        with torch.no_grad():
            y_pred = self.model(
                x_valid
            )

            loss = self.loss_fn(
                y_pred,
                y_valid,
            )

        return loss.item()