import numpy as np
import torch
import torch.nn.functional as F
import librosa


class Normaliser:
    def fit(self, x):
        if x.dtype == np.object:
            xl = np.vstack(x)
            self.mins = xl.min(axis=0)
            self.maxs = xl.max(axis=0)
        else:
            self.mins = x.min(axis=0)
            self.maxs = x.max(axis=0)

    def fit_transform(self, x):
        if x.dtype == np.object:
            xl = np.vstack(x)
            xn = np.empty_like(x)
            self.mins = xl.min(axis=0)
            self.maxs = xl.max(axis=0)
            for i, xi in enumerate(x):
                xn[i] = np.zeros_like(xi)
                for j in range(xi.shape[-1]):
                    xn[i][:, j] = (xi[:, j] - self.mins[j]) / (
                        self.maxs[j] - self.mins[j]
                    )
        else:
            xn = np.copy(x)
            self.mins = x.min(axis=0)
            self.maxs = x.max(axis=0)
            for j in range(x.shape[-1]):
                xn[:, j] = (xn[:, j] - self.mins[j]) / (self.maxs[j] - self.mins[j])
        return xn

    def transform(self, x):
        if x.dtype == np.object:
            xn = np.empty_like(x)
            for i, xi in enumerate(x):
                xn[i] = np.zeros_like(xi)
                for j in range(xi.shape[-1]):
                    xn[i][:, j] = (xi[:, j] - self.mins[j]) / (
                        self.maxs[j] - self.mins[j]
                    )
        else:
            xn = np.copy(x)
            for j in range(x.shape[-1]):
                xn[:, j] = (xn[:, j] - self.mins[j]) / (self.maxs[j] - self.mins[j])
        return xn


def create_1hot_like(Nout, x, labels):
    y1h = np.empty_like(x)
    for i, l in enumerate(labels):
        y1h[i] = np.zeros((x[i].shape[0], Nout))
        y1h[i][:, int(l)] = 1.0
    return y1h


def convert_2d_to_nested_1d(arr: np.ndarray, nested_dim: int) -> np.ndarray:
    """Convert a 2D NumPy array to a 1D array with nested 2D arrays"""
    n_digits = len(arr)
    one_d_arr = np.empty((n_digits,), dtype=np.object)

    for i in range(n_digits):
        one_d_arr[i] = arr.reshape(n_digits, nested_dim, nested_dim)[i, :, :]

    return one_d_arr


def torch_one_hot(label_array: np.ndarray) -> np.ndarray:
    """Return a numpy one hot encoded array using the built in Pytorch functionality"""
    one_h = F.one_hot(torch.tensor(label_array))
    return one_h.numpy()


def mfcc_preprocessing(signals: np.ndarray, rates: np.ndarray) -> np.ndarray:
    x = np.empty_like(signals)

    for i, (sig, rate) in enumerate(zip(signals, rates)):
        signal_vals = sig.astype(float)
        x[i] = librosa.feature.mfcc(
            y=signal_vals,
            sr=rate,
            n_fft=512,
            hop_length=256,
            lifter=26,
            n_mfcc=13,
            power=1,
        )
        x[i] = x[i].T

    return x
