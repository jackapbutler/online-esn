import os
from typing import Tuple

import numpy as np
import sklearn.model_selection as sk_mod
import utils


def load_mnist():
    dirpath = os.path.join(os.path.dirname(__file__), "mnist", "mnist_data.npz")

    with np.load(dirpath) as data:
        train_imgs = data["train_imgs"]
        train_labels = data["train_labels"]
        test_imgs = data["test_imgs"]
        test_labels = data["test_labels"]

    return train_imgs, train_labels, test_imgs, test_labels


def load_and_process_mnist_col_by_col(
    reduce_size: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load and transform the MNIST dataset in a column by column format"""
    x_train, l_train, x_test, l_test = load_mnist()

    if reduce_size:
        # Limit to only 20% of training set size due to hardware constraints
        x_train, _, l_train, _ = sk_mod.train_test_split(
            x_train,
            l_train,
            test_size=0.80,
            stratify=[int(np.argmax(l)) for l in l_train],
        )

    N_COLS = 28
    N_LABS = len(np.unique(l_train))

    train_col_by_col = utils.convert_2d_to_nested_1d(x_train, N_COLS)
    test_col_by_col = utils.convert_2d_to_nested_1d(x_test, N_COLS)

    y_train, y_test = (
        np.copy(l_train.flatten()).astype(np.int64),
        np.copy(l_test.flatten()).astype(np.int64),
    )
    y_train_labels = np.repeat(utils.torch_one_hot(y_train), repeats=N_COLS, axis=0)
    y_test_labels = np.repeat(utils.torch_one_hot(y_test), repeats=N_COLS, axis=0)

    return (
        train_col_by_col,
        y_train_labels.reshape(len(y_train), N_COLS, N_LABS),
        test_col_by_col,
        y_test_labels.reshape(len(y_test), N_COLS, N_LABS),
        N_LABS,
    )
