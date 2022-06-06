import os
from typing import List, Tuple

import numpy as np
import src.datasets.utils as utils

SPEAKERS_SUBSET = ["f1", "f2", "f3", "f4", "f5"]  # to replicate previous paper


def load_TI20(speakers=None, digits_only=True, train=True, rngseed=None):

    dirpath = os.path.join(os.path.dirname(__file__), "ti46")

    subset = "train" if train else "test"

    datafile = np.load(
        os.path.join(dirpath, "TI20_" + subset + "_data.npz"), allow_pickle=True
    )

    signal = datafile[subset + "_signal"]
    label = datafile[subset + "_label"]
    speaker = datafile[subset + "_speaker"]
    rate = datafile[subset + "_rate"]

    speakers = np.array(speakers) if speakers is not None else np.unique(speaker)

    labels = np.arange(0, 10) if digits_only else np.unique(label)

    inds = np.array([], dtype=int)
    for s in speakers:
        for lab in labels:
            inds = np.append(inds, np.where(np.logical_and(speaker == s, label == lab)))

    rng = np.random.default_rng(seed=rngseed)
    rng.shuffle(inds)

    return signal[inds], label[inds], rate[inds], speaker[inds]


def load_and_process_ti46(
    speakers: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load and transform the Ti46 dataset for a certain range of speakers using the MFCC transformation.
    This is loaded in a form where each sample is actually made of multiple sub-samples (due to time multiplexing).
    All sub-samples must be passed and the aggregate popular prediction taken as the model's decision (like passing Mnist column by column).
    """
    if not speakers:
        # default to common subset
        speakers = SPEAKERS_SUBSET

    # Load the datasets for training and testing (train=False returns more samples)
    (
        train_signal,
        train_label,
        train_rate,
        _,
    ) = load_TI20(speakers, train=False)
    test_signal, test_label, test_rates, _ = load_TI20(speakers)

    # Apply a audio signal preprocessing step called MFCC
    x_train, x_test = (
        utils.mfcc_preprocessing(train_signal, train_rate),
        utils.mfcc_preprocessing(test_signal, test_rates),
    )

    # Normalise the output of MFCC
    prescaler = utils.Normaliser()
    xn_train, xn_test = (
        prescaler.fit_transform(x_train),
        prescaler.fit_transform(x_test),
    )

    # create one-hot encoding (due to time multiplexing)
    y_train_1h, y_test_1h = (
        utils.create_1hot_like(len(np.unique(train_label)), xn_train, train_label),
        utils.create_1hot_like(len(np.unique(test_label)), xn_test, test_label),
    )

    return (xn_train, y_train_1h, xn_test, y_test_1h, len(np.unique(train_label)))
