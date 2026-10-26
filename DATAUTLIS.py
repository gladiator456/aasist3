"""ASVspoof 2019 Logical Access protocol and audio dataset utilities."""
import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

TARGET_SAMPLES = 64600
TARGET_SAMPLE_RATE = 16000


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    """Parse protocol rows: speaker_id utterance_id - system_id bonafide|spoof."""
    if not os.path.isfile(dir_meta):
        raise FileNotFoundError(f"Protocol file not found: {dir_meta}")
    labels, file_list = {}, []
    with open(dir_meta, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Malformed protocol row {line_no} in {dir_meta}: {line.strip()}")
            _, key, _, _, label = parts
            file_list.append(key)
            if not is_eval:
                if label not in {"bonafide", "spoof"}:
                    raise ValueError(f"Unknown label {label!r} on protocol line {line_no}")
                labels[key] = 1 if label == "bonafide" else 0
    return file_list if is_eval else (labels, file_list)


def _mono_float32(audio, path):
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"Audio is empty or has unsupported shape {audio.shape}: {path}")
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return audio


def pad(x, max_len=TARGET_SAMPLES):
    """Deterministic crop or tile; safe for empty inputs."""
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Cannot pad an empty audio signal.")
    if x.shape[0] >= max_len:
        return x[:max_len]
    return np.tile(x, int(np.ceil(max_len / x.shape[0])))[:max_len]


def pad_random(x, max_len=TARGET_SAMPLES):
    """Random crop for long training clips; tile short clips."""
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Cannot pad an empty audio signal.")
    if x.shape[0] >= max_len:
        start = np.random.randint(0, x.shape[0] - max_len + 1)
        return x[start:start + max_len]
    return np.tile(x, int(np.ceil(max_len / x.shape[0])))[:max_len]


def apply_pre_emphasis(x, coef=0.97):
    x = np.asarray(x)
    return x - coef * np.concatenate(([0.0], x[:-1]))


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, is_train=True, sample_rate=TARGET_SAMPLE_RATE):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.is_train = is_train
        self.sample_rate = sample_rate
        self.cut = TARGET_SAMPLES
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Audio directory not found: {base_dir}")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        path = os.path.join(self.base_dir, f"{key}.flac")
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sr} Hz: {path}. Resample before training.")
        audio = _mono_float32(audio, path)
        audio = pad_random(audio, self.cut) if self.is_train else pad(audio, self.cut)
        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        return waveform, int(self.labels[key])


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir, sample_rate=TARGET_SAMPLE_RATE):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.sample_rate = sample_rate
        self.cut = TARGET_SAMPLES
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Audio directory not found: {base_dir}")

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        path = os.path.join(self.base_dir, f"{key}.flac")
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sr} Hz: {path}")
        audio = pad(_mono_float32(audio, path), self.cut)
        return torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), key
