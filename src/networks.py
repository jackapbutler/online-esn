"""Module with a collection of Pytorch Neural Network model architecture which can be used for classification or regression."""
import torch


class BaseLinear(torch.nn.Module):
    """A Linear Network of weights with bias terms"""

    def __init__(self, Nin: int, Nout: int, use_bias: bool = True) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(Nin, Nout, bias=use_bias))

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the linear layer"""
        return self.layers(xb)


class SpaRCe(torch.nn.Module):
    """
    Learning adaptive neuronal thresholds for sparse representations.
    From the following paper: https://arxiv.org/pdf/1912.08124.pdf
    """

    def __init__(self, Nin: int, Nout: int, thetas: torch.Tensor):
        super().__init__()
        # quicker in single precision
        self.linear = torch.nn.Linear(Nin, Nout).double()
        self.thetas = torch.nn.Parameter(data=thetas, requires_grad=True)
        self.activation = torch.nn.ReLU()

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the SpaRCe neural network."""
        activation_status = self.activation(torch.abs(xb) - self.thetas)
        output_effect = self.linear(torch.sign(xb) * activation_status)
        return output_effect


class TwoLayerMlp(torch.nn.Module):
    """A Two Layer Linear Network with bias terms and ReLU non-linearity"""

    def __init__(self, Nin: int, Nout: int, use_bias: bool = True, **kwargs) -> None:
        super().__init__()
        self.Nhid = kwargs.get("Nhid", Nin)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(Nin, self.Nhid, bias=use_bias),
            torch.nn.ReLU(),
            torch.nn.Linear(self.Nhid, Nout, bias=use_bias),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the MLP"""
        return self.layers(xb)
