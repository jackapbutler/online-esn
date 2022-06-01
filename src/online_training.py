"""Module for testing and training online Pytorch linear classification models"""
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as sk_mod
import torch
import networks as nets
import torch.nn.functional as F
import torch.utils.data as torch_data


@dataclass
class LossCriterion:
    """Loss criterion dataclass for Pytorch"""

    def __init__(
        self,
        function=F.cross_entropy,
        l1_lambda: float = 0.2,
        l2_lambda: float = 0.5,
    ):
        self.function = function
        self.use_l1 = bool(l1_lambda)
        self.use_l2 = bool(l2_lambda)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda


OPTIMISERS = {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam}

LOSS_CRITERION = {"CEL": F.cross_entropy}


class Classifier(torch.nn.Module):
    """Base online learning linear classification class for Pytorch"""

    def __init__(self, Nin: int, Nout: int, model=None):
        """Initialise the network the number of in/out dimensions (features)"""
        super(Classifier, self).__init__()

        self.model = model
        if not self.model:
            self.model = nets.BaseLinear(Nin, Nout)

    def forward(self, x):
        """Do a forward pass through the network"""
        return self.model(x)

    def count_params(self) -> Dict[str, int]:
        """
        Count the number of trainable & non-trainable parameters in the model.
        Returns a Dict containing the counts for trainable, non-trainable and total parameters.
        """
        non_trainable = 0
        trainable = 0

        for p in self.model.parameters():
            if p.requires_grad:
                trainable += p.numel()
            else:
                non_trainable += p.numel()

        return {
            "total": trainable + non_trainable,
            "trainable": trainable,
            "non-trainable": non_trainable,
        }

    def prepare_dataloaders(
        self,
        x_train: np.ndarray,
        x_valid: np.ndarray,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        bs: int,
    ) -> Tuple[torch_data.DataLoader, torch_data.DataLoader]:
        """Prepares Pytorch training & validation DataLoaders for model training"""
        x_train, y_train, x_valid, y_valid = map(
            torch.tensor, (x_train, y_train, x_valid, y_valid)
        )

        train_ds = torch_data.TensorDataset(x_train, y_train)
        valid_ds = torch_data.TensorDataset(x_valid, y_valid)

        self.train_dl = torch_data.DataLoader(train_ds, batch_size=bs, shuffle=True)
        self.valid_dl = torch_data.DataLoader(valid_ds, batch_size=bs * 2)

    def get_validation_eval(self, loss_crit) -> Tuple[float, float]:
        """Get the current validation loss and accuracy for the model"""
        with torch.no_grad():
            valid_loss = 0.0
            valid_acc = 0
            for xb, yb in self.valid_dl:
                pred = self.model(xb)
                valid_loss += loss_crit.function(pred, yb).item()
                valid_acc += accuracy(pred, yb).item()

        return valid_loss / len(self.valid_dl), valid_acc / len(self.valid_dl)

    def fit(
        self,
        max_epochs: int,
        loss_critierion: LossCriterion,
        optimiser,
        verbose: bool = False,
        parameter_save_path: str = None,
    ) -> np.ndarray:
        """Fits a linear model to training data using Pytorch DataLoaders and EarlyStopping"""
        train_history = np.zeros((5, max_epochs))
        PATIENCE = 10  # epochs to wait without validation improvement

        patience = 10
        old_valid_loss, old_valid_acc = self.get_validation_eval(loss_critierion)

        for epoch in range(max_epochs):
            if parameter_save_path:
                np.savetxt(
                    f"{parameter_save_path}/{epoch}", self.model.thetas.data.numpy()
                )

            self.model.train()
            train_loss = 0.0
            train_acc = 0.0

            for xb, yb in self.train_dl:
                # get predictions
                pred = self.model(xb)

                # get loss
                loss: torch.Tensor = loss_critierion.function(pred, yb)
                train_loss += loss.item()
                train_acc += accuracy(pred, yb).item()

                # add regularisation
                if loss_critierion.use_l1 or loss_critierion.use_l2:
                    params = self.fetch_parameters()

                    if loss_critierion.use_l1:
                        loss += loss_critierion.l1_lambda * self.compute_l1_loss(params)
                    if loss_critierion.use_l2:
                        loss += loss_critierion.l2_lambda * self.compute_l2_loss(params)

                # propagate loss backwards
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

            train_loss /= len(self.train_dl)
            train_acc /= len(self.train_dl)

            self.model.eval()
            curr_valid_loss, curr_valid_acc = self.get_validation_eval(loss_critierion)

            if verbose:
                print(epoch + 1, train_loss, train_acc, curr_valid_loss, curr_valid_acc)

            train_history[:, epoch] = np.array(
                [epoch, train_loss, curr_valid_loss, train_acc, curr_valid_acc]
            )

            # early stopping clauses
            if self.early_stopping(old_valid_loss, curr_valid_loss):
                patience -= 1
            else:
                patience = PATIENCE

            if patience == 0:
                print(f"Stopping early at at end of epoch {epoch+1}")
                break

        return train_history

    def training_session(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Run a training session using the `fit()` method with the specified parameters"""
        self.prepare_dataloaders(
            x_train,
            x_valid,
            y_train,
            y_valid,
            bs=kwargs.get("batch_size", 100),
        )

        return self.fit(
            max_epochs=kwargs.get("max_epochs", 201),
            loss_critierion=LossCriterion(
                function=LOSS_CRITERION[kwargs.get("criterion", "CEL")],
                l1_lambda=kwargs.get("l1_lambda", 0),
                l2_lambda=kwargs.get("l2_lambda", 0),
            ),
            optimiser=OPTIMISERS[kwargs.get("optimiser", "Adam")](
                params=self.parameters(), lr=kwargs.get("lr", 0.04)
            ),
            parameter_save_path=kwargs.get("path"),
        )

    def fetch_parameters(self) -> torch.Tensor:
        """Collect the model parameters for regularisation"""
        reg_parameters = []

        for parameter in self.parameters():
            reg_parameters.append(parameter.view(-1))

        return torch.cat(reg_parameters)

    @staticmethod
    def prepare_training_data(
        x_train: np.ndarray,
        y_train_1h: np.ndarray,
        x_test: np.ndarray,
        y_test_1h: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split and vertically stack the input & output matrices into training, validation and testing sets.
        Note: This method uses stratified sampling of the input labels.
        """
        (
            z_split_train,
            z_split_valid,
            y_split_train,
            y_split_valid,
        ) = sk_mod.train_test_split(
            x_train,
            y_train_1h,
            test_size=0.25,
            shuffle=True,
            stratify=[int(np.argmax(y)) for y in y_train_1h],
            random_state=1,
        )

        # we don't want to stack the test data (as this may contain time step breakdowns)
        (z_train_flat, z_valid_flat,) = (
            np.vstack(z_split_train),
            np.vstack(z_split_valid),
        )

        y_train = np.array([np.argmax(x) for x in np.vstack(y_split_train)])
        y_valid = np.array([np.argmax(x) for x in np.vstack(y_split_valid)])
        y_test = np.array([np.argmax(x) for x in np.vstack(y_test_1h)])

        return (
            z_train_flat,
            y_train,
            z_valid_flat,
            y_valid,
            x_test,
            y_test,
            z_split_train,
            y_split_train,
        )

    @staticmethod
    def early_stopping(old_valid_loss: float, curr_valid_loss: float) -> bool:
        """Decides if the new validation loss violates an early stopping condition"""
        should_stop = False
        loss_diff = curr_valid_loss - old_valid_loss

        if 0 < loss_diff:
            should_stop = True

        return should_stop

    @staticmethod
    def compute_l1_loss(parameters: torch.Tensor) -> torch.Tensor:
        """Compute the L1 Regularisation loss at a particular step"""
        return torch.abs(parameters).sum()

    @staticmethod
    def compute_l2_loss(parameters: torch.Tensor) -> torch.Tensor:
        """Compute the L2 Regularisation loss at a particular step"""
        return torch.square(parameters).sum()

    def predict(self, x: np.ndarray) -> int:
        """
        Predict the label for a new sample or array of multiple time step samples.
        """
        with torch.no_grad():
            if 1 < x.ndim:
                # if we have multiple time steps we need to aggregate predictions
                preds = []
                for sample in x:
                    input = torch.autograd.Variable(torch.from_numpy(sample).double())
                    preds.append(int(np.argmax(self.model(input))))
                pred = max(set(preds), key=preds.count)

            else:
                # we have a no time steps and can predict directly
                input = torch.autograd.Variable(torch.from_numpy(x).double())
                pred = int(np.argmax(self.model(input)))

        return pred


def accuracy(out: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """Calculates the accuracy for a given training epoch"""
    preds = torch.argmax(out, dim=1)
    true = torch.argmax(yb, dim=1) if len(yb.shape) > 1 else yb
    return (preds == true).float().mean()


def plot_results_array(results_arr: np.ndarray):
    """Plot the results array generated during model training session"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle("Validation Accuracy and Loss over Model Training")

    ax1.plot(results_arr[0], results_arr[1], label="training")
    ax1.plot(results_arr[0], results_arr[2], label="validation")
    ax1.set(xlabel="Epochs", ylabel="Loss")

    ax2.plot(results_arr[0], results_arr[3], label="training")
    ax2.plot(results_arr[0], results_arr[4], label="validation")
    ax2.set(xlabel="Epochs", ylabel="Accuracy %")

    plt.legend(["Training", "Validation"])
    plt.show()


def plot_results_json(result_json: Dict):
    """
    Plot the train_history of online regression training
    (needs to be passed a dictionary of the same format as in `experiment_helpers.ResultsHandler.generate_results_dict())`
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(
        f"Data: {result_json['dataset']}, Reservoir: {result_json['reservoir']['method']}, Hidden: {result_json['hidden neurons']}, Loss: {result_json['output layer']['configuration']['criterion']}"
    )

    ax1.plot(result_json["history"]["training loss"], label="training")
    ax1.plot(result_json["history"]["validation loss"], label="validation")
    ax1.set(xlabel="Epochs", ylabel="Loss")
    ax1.legend()

    ax2.plot(result_json["history"]["training accuracy"], label="training")
    ax2.plot(result_json["history"]["validation accuracy"], label="validation")
    ax2.set(xlabel="Epochs", ylabel="Accuracy%")
    ax2.legend()

    plt.show()


def torch_summarize(
    model: torch.nn.Sequential, show_weights=True, show_parameters=True
) -> str:
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + " (\n"
    for key, module in model._modules.items():
        modstr = torch_summarize(module)
        modstr = torch.nn.modules.module._addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += "  (" + key + "): " + modstr
        if show_weights:
            tmpstr += ", weights={}".format(weights)
        if show_parameters:
            tmpstr += ", parameters={}".format(params)
        tmpstr += "\n"

    tmpstr = tmpstr + ")"
    return tmpstr
