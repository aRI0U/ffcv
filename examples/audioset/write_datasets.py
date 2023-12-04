import os

import numpy as np
import torch.testing
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from ffcv.loader import Loader, OrderOption
from ffcv.fields import SpectrogramField
from ffcv.fields.decoders import SpectrogramDecoder
from ffcv.transforms import ToDevice, ToTensor
from ffcv.writer import DatasetWriter


def make_collate(timesteps):
    def collate_fn(spectrograms):
        batch = []
        for s, in spectrograms:
            rd_idx = np.random.randint(s.shape[-1] - timesteps)
            batch.append(s[..., rd_idx: rd_idx + timesteps])
        return (torch.from_numpy(np.concatenate(batch)),)
    return collate_fn


class RecursiveDataset(Dataset):
    def __init__(self, root_dir, ext):
        self.root_dir = root_dir
        self.ext = ext
        self.npy_files = self._find_files()

    def _find_files(self):
        npy_files = []
        for root, _, files in tqdm(os.walk(self.root_dir)):
            for file in files:
                if file.endswith(self.ext):
                    npy_files.append(os.path.join(root, file))
        return npy_files

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_file = self.npy_files[idx]
        return (np.load(npy_file)[None, :, :],)


def main(data_path, extension):
    # consts
    batch_size = 256
    num_workers = 16
    timesteps = 208

    # create initial dataset and corresponding data loader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = RecursiveDataset(data_path, extension)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=make_collate(timesteps),
        shuffle=False,
        drop_last=True
    )

    print(f"Writing {data_path}.beton...")
    writer = DatasetWriter(data_path + '.beton', {
        "audio": SpectrogramField()
    })
    # writer.from_indexed_dataset(dataset)

    fields = {
        "audio": SpectrogramField
    }
    pipelines = {
        "audio": [SpectrogramDecoder(timesteps), ToTensor()]#, ToDevice(device)]
    }

    ffcv_loader = Loader(
        data_path + '.beton',
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,
        pipelines=pipelines,
        custom_fields=fields
    )

    if False:
        a_list, b_list = [], []
        for i, (a,) in enumerate(loader):
            a_list.append(a)#.to(device))

        for i, (b,) in enumerate(ffcv_loader):
            b_list.append(b.clone())

        for i, a, b in zip(range(len(a_list)), a_list, b_list):
            if a.ndim == 5:
                a = a.squeeze(0)
            if a.size(-1) == b.size(-1):
                torch.testing.assert_close(a, b)
            elif a.size(-1) < b.size(-1):
                torch.testing.assert_close(a, b[..., :a.size(-1)])
            else:
                for j in range(a.size(-1)):
                    try:
                        torch.testing.assert_close(a[..., j:j+b.size(-1)], b)
                    except AssertionError:
                        continue
                    else:
                        print(j)
                        break
                if j + 1 == a.size(-1):
                    print(a, a.shape)
                    print(b, b.shape)
                    raise IndexError
                print(i, 'passed')

    print("Iterating over Pytorch DataLoader...")
    # for batch, in tqdm(loader):
    #     continue
    #     batch = batch.to(device)

    print("Iterating over FFCV DataLoader...")
    for batch, in tqdm(ffcv_loader):
        continue
        assert batch.shape == (batch_size, 1, 80, timesteps), str(batch.shape)


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "/home/data3/alain/m2d/data/audioset_lms/unbalanced_train"
    extension = sys.argv[2] if len(sys.argv) > 2 else ".npy"

    main(data_path, extension)
