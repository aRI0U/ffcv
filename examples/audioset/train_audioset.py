from tqdm import tqdm

from torch.utils.data import DataLoader

from ffcv.loader import Loader, OrderOption
from ffcv.fields import SpectrogramField
from ffcv.fields.decoders import SpectrogramDecoder
from ffcv.transforms import ToTensor


def main(data_path):
    fields = {
        "audio": SpectrogramField
    }
    pipelines = {
        "audio": [SpectrogramDecoder((80, 1001)), ToTensor()]
    }

    ffcv_loader = Loader(
        data_path,
        batch_size=1,
        num_workers=0,
        order=OrderOption.SEQUENTIAL,
        pipelines=pipelines,
        custom_fields=fields
    )

    for i, (batch,) in tqdm(enumerate(ffcv_loader), total=len(ffcv_loader)):
        print(batch.squeeze(), type(batch), batch.dtype, batch.shape)
        if i == 5:
            break



if __name__ == "__main__":
    import sys

    main("/home/alain/datasets/AudioSet/mel.beton")
