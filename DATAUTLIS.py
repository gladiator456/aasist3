import os
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

__author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    """
    Reads an ASVspoof2019 LA protocol file.
    Line format: SPEAKER_ID AUDIO_FILE_NAME ENV SYSTEM_ID KEY
    """
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    """Deterministic pad/crop — used for dev/eval so scores are reproducible."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def pad_random(x: np.ndarray, max_len: int = 64600):
    """Random crop — used for training, acts as data augmentation."""
    x_len = x.shape[0]
    if x_len >= max_len:
        stt = np.random.randint(0, x_len - max_len + 1)
        return x[stt:stt + max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def apply_pre_emphasis(x, coef=0.97):
    """x_l = x_l - 0.97 * x_{l-1}"""
    return x - coef * np.append(0, x[:-1])


class Dataset_ASVspoof2019_train(Dataset):
    """
    Works for BOTH the train split and the dev split, since both have labels.
    base_dir must point directly at the folder containing the .flac files
    (e.g. .../ASVspoof2019_LA_train/flac).
    """
    def __init__(self, list_IDs, labels, base_dir, is_train=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.is_train = is_train
        self.cut = 64600  # ~4 sec at 16kHz

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        path = os.path.join(self.base_dir, f"{key}.flac")
        X, _ = sf.read(path)

        X_pad = pad_random(X, self.cut) if self.is_train else pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    """For blind eval protocols that have no labels."""
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        path = os.path.join(self.base_dir, f"{key}.flac")
        X, _ = sf.read(path)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
