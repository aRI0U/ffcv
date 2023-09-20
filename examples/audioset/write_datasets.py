import os

import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset

from ffcv.writer import DatasetWriter
from ffcv.fields import SpectrogramField


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
    dataset = RecursiveDataset(data_path, extension)

    writer = DatasetWriter(data_path + '.beton', {
        "audio": SpectrogramField()
    })
    writer.from_indexed_dataset(dataset)


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "/home/alain/datasets/AudioSet/mel"
    extension = sys.argv[2] if len(sys.argv) > 2 else ".npy"

    main(data_path, extension)
