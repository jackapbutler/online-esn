"""
This module implements an single Echo State Neural Network.
"""
import copy
from typing import Tuple

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


class ESN:
    """
    A Single Echo State Network with `nIn` input weights and `nRes` reservoir neurons.
    The `ESN.transform` method will perform a transformation of sequential input data using the reservoir.
    """

    def __init__(self, nIn: int, nRes: int, **kwargs):
        self.nIn: int = nIn
        self.nRes: int = nRes

        self.alpha: float = kwargs.get("alpha", 0.90)
        self.gamma: float = kwargs.get("gamma", 0.01)
        self.rho: float = kwargs.get(
            "rho", 0.98
        )  # NOTE can try removing this rescaling
        self.sparsity: float = kwargs.get("sparsity", 0.90)
        self.batch_reset: bool = kwargs.get("batch_reset", False)

        self.act = kwargs.get("activation", torch.tanh)

        self.Win: torch.Tensor = self.initialise_input_weights(nIn, nRes, self.gamma)
        self.Wres: torch.Tensor = self.generate_reservoir_weights(
            nRes, self.sparsity, self.rho
        )
        self.reset_states()

    def reset_states(self):
        """Reset the hidden states of the ESN to zero"""
        self.hidden = torch.zeros([1, self.nRes])

    @staticmethod
    def initialise_input_weights(
        nIn: int, reservoir_nodes: int, gamma: float
    ) -> torch.Tensor:
        """
        Generate a [nIn, reservoir_nodes] matrix of reservoir weights
        and scale this weights using a `gamma` factor.

        Args:
            nIn: Number of input data dimensions.
            reservoir_nodes: Number of internal reservoir nodes.
            gamma: Scaling factor for the input weight matrix (affects contribution of input data to states).

        Returns:
            A [nIn, reservoir_nodes] tensor of input weights to the reservoir nodes.
        """
        input_weights = gamma * np.random.randn(nIn, reservoir_nodes)
        return torch.tensor(input_weights)

    @staticmethod
    def generate_reservoir_weights(
        reservoir_nodes: int, sparsity: float, rho: float
    ) -> torch.Tensor:
        """
        Generate a [reservoir_nodes, reservoir_nodes] matrix of reservoir weights
        with given `sparsity` and scale this weights using a `rho` factor.

        Args:
            reservoir_nodes: Number of internal reservoir nodes.
            sparsity: Proportion of sparsity within the reservoir [0,1].
            rho: Scaling factor for the reservoir weights (affects contribution of previous reservoir states).

        Returns:
            A sparse [reservoir_nodes, reservoir_nodes] tensor of weights between reservoir nodes.
        """
        # randomly zero a matrix, entries in range [-1,1]
        W = np.random.uniform(-1, 1, (reservoir_nodes, reservoir_nodes))

        num_zeros = np.ceil(sparsity * reservoir_nodes).astype(int)
        for col in range(reservoir_nodes):
            row_indices = np.random.permutation(reservoir_nodes)
            zero_indices = row_indices[:num_zeros]
            W[zero_indices, col] = 0

        # apply scaling factor
        Wres = rho * W / (np.max(np.absolute(np.linalg.eigvals(W))))
        return torch.tensor(Wres)

    def update_hidden_states(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate the new hidden state of the ESN. The new hidden states are a
        linear combination of previous input and the dynamics of the reservoir.

        HiddenState(t+1) = (1-alpha) * HiddenState(t) + alpha * f [ gamma * W_input * Input(t) + rho * W_reservoir * HiddenState(t) ]

        * `f` is a non-linear function such as `tanh()`.
        * `gamma` and `rho` effects are applied to the weights during initialisation.

        HiddenState(t+1) = (1-alpha) * HiddenState(t) + alpha * f [ W_input * Input(t) + W_reservoir * HiddenState(t) ]

        Args:
            s_input: The input signal for a particular sample.

        Returns:
            s_output: The transformed input signal.
        """
        s_input = torch.tensor(signal).unsqueeze(0)

        # Calculate the previous state, input data & reservoir weights
        # influence on the transformation of the input data.
        state = copy.copy(self.hidden[-1]).unsqueeze(0)
        data = torch.matmul(s_input, self.Win)
        reservoir = torch.matmul(state, self.Wres)

        s_output = (1 - self.alpha) * state + self.alpha * self.act(data + reservoir)

        # add newly calculated internal reservoir hidden state
        self.hidden = torch.cat((self.hidden, s_output), dim=0)

        return s_output.numpy()

    def transform(
        self, x_input: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Transform observations using the ESN and update the reservoir hidden states.

        Args:
            x_input: A NumPy array containing a batch of training data samples.
            The input shape should be [# Samples,] with each sample having a shape of [Any, # Input Timesteps].

        Returns:
            A NumPy array which contains transformed input data.
            A Pytorch Tensor containing the historical hidden states.

        Raises:
            ValueError: All input samples must have the same dimensions and be [Any, N_Input].
        """
        shape_violation = [x.shape[-1] != self.nIn for x in x_input]

        if any(shape_violation):
            raise ValueError(
                f"All timesteps for each input sample should have a shape of size {self.nIn}, found {shape_violation.count(True)} violations."
            )

        else:
            # transform each sample in the input batch
            output = []
            for sample in x_input:

                # transform each timestep in the sample
                transformed_sample = []
                for timestep in sample:
                    transformed_sample.append(self.update_hidden_states(timestep))

                sample_arr = np.array(transformed_sample, dtype=float)
                output.append(sample_arr.reshape(sample.shape[0], self.nRes))

            if self.batch_reset:
                # reset states after each sample
                self.reset_states()

            return np.array(output), self.hidden
