import glob
import os
import random
import shutil
import time

import scipy.io.wavfile as wavfile
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ffcv.fields import IntField
from ffcv.fields.decoders import IntDecoder
from ffcv.fields.waveform import WaveformField, WaveformDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice
from ffcv.transforms.common import Squeeze
from ffcv.transforms.audio import Int16ToFloat
from ffcv.writer import DatasetWriter


class DummyDataset(Dataset):
    def __init__(self, path, dtype='<f4'):
        self.paths = sorted(glob.glob(f"{path}/**/*.wav", recursive=True))
        self.dtype = dtype

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x, sr = self.load_audio(self.paths[idx])
        return x.numpy(), idx % 10

    def load_audio(self, path: str):
        sr, x = wavfile.read(path)
        print(x)
        return torch.from_numpy(x.T), sr


class DummyCNN(nn.Module):
    def __init__(self, chans: int = 1, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chans, num_classes // 2, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv1d(num_classes // 2, num_classes // 2, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_classes // 2, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


def build_dataset(n_elems, h, w, sr=16000, dtype='<f4'):
    height = torch.arange(h, dtype=torch.float32).view(h, 1)

    os.makedirs("tmp", exist_ok=True)

    for i in range(n_elems):
        # width = torch.arange(random.randint(int(sr * w), int(2 * sr *w)), dtype=torch.float32).view(1, -1)
        width = torch.arange(random.randint(w, 2 * w), dtype=torch.float32).view(1, -1)
        tensor = height / 100 + width / 1000 + i / 10
        if dtype == '<i2':
            tensor = -tensor.mul(2**15).to(torch.int16)
        print(tensor)
        wavfile.write(f"tmp/{i}.wav", sr, tensor.numpy().T)

    return DummyDataset("tmp", dtype=dtype)


def to_beton(dataset, path, dtype):
    writer = DatasetWriter(path, {"audio": WaveformField(dtype=dtype), "label": IntField()})
    writer.from_indexed_dataset(dataset)


def build_loader(path, device, timesteps=50, batch_size=2, num_workers=0, dtype='<f4'):
    fields = {
        "audio": WaveformField,
        "label": IntField
    }
    pipelines = {
        "audio": [WaveformDecoder(timesteps, num_channels=channels, dtype=dtype), ToTensor(), ToDevice(device, non_blocking=True)],
        "label": [IntDecoder(), ToTensor(), ToDevice(device, non_blocking=True), Squeeze(1)],
    }

    if dtype == '<i2':
        pipelines["audio"].append(Int16ToFloat())

    return Loader(
        path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.QUASI_RANDOM,
        pipelines=pipelines,
        custom_fields=fields
    )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = '<i2'
    debug = True

    dataset_path = "tmp.beton"
    channels = 2
    num_classes = 10

    if debug:
        kwargs = dict(timesteps=8, batch_size=2, num_workers=0)
        ds = build_dataset(2, channels, 9, dtype=dtype)

    else:
        kwargs = dict(timesteps=51, batch_size=10, num_workers=8)
        ds = build_dataset(10, channels, 67, dtype=dtype)

    to_beton(ds, dataset_path, dtype=dtype)
    shutil.rmtree("tmp/")

    loader = build_loader(dataset_path, device, **kwargs, dtype=dtype)

    if debug:
        for x, y in loader:
            print(x, y, x.shape)
        for x, y in loader:
            print(x, y)
        exit(0)

    # do dummy training
    net = DummyCNN(chans=channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss = None

    t0 = time.time()

    pbar = trange(1 if debug else 100, leave=True)
    for _ in pbar:
        for x, y in loader:
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        pbar.set_description(f'loss: {loss.cpu().item():.3f}')
    t1 = time.time()

    print("Time elapsed", t1 - t0)