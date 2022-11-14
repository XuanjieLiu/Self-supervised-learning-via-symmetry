from torchvision.datasets.utils import download_url, check_integrity
from torchvision import transforms
from PIL import Image
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


MIN_SEQ_LEN = 12

def cut_data_seq(seq_data, min_len):
    selected_data = list(filter(lambda x: len(x) >= min_len, seq_data))
    cut_data = [data[0:min_len] for data in selected_data]
    tensor_data = torch.tensor(cut_data, dtype=torch.float)
    return tensor_data


class KittiMasks(Dataset):
    """
	latents encode:
	0: center of mass vertical position
	1: center of mass horizontal position
	2: area
	"""

    def __init__(self, path='./data/kitti/', transform=None,
                 max_delta_t=5, seq_len=MIN_SEQ_LEN):
        self.path = path
        self.data = None
        self.latents = None
        self.lens = None
        self.cumlens = None
        self.max_delta_t = max_delta_t
        # self.url = 'https://zenodo.org/record/3931823/files/kitti_peds_v2.pickle?download=1'
        self.seq_len = seq_len

        if transform == 'default':
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomAffine(degrees=(2., 2.), translate=(5 / 64., 5 / 64.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    lambda x: x.numpy()
                ])
        else:
            self.transform = None

        self.load_data()

    def load_data(self):
        # download if not avaiable
        file_path = self.path
        # if not os.path.exists(file_path):
        #     os.makedirs(self.path, exist_ok=True)
        #     print(f'file not found, downloading from {self.url} ...')
        #     from urllib import request
        #     url = self.url
        #     request.urlretrieve(url, file_path)

        with open(file_path, 'rb') as data:
            data = pickle.load(data)

        self.data = cut_data_seq(data['pedestrians'], self.seq_len).unsqueeze(2)
        self.latents = cut_data_seq(data['pedestrians_latents'], self.seq_len)

        self.lens = [len(seq) - 1 for seq in self.data]  # start image in sequence can never be starting point
        self.cumlens = np.cumsum(self.lens)

    def __getitem__(self, index):
        return self.data[index], self.latents[index]



    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = KittiMasks()
    loader = DataLoader(dataset, batch_size=32)
    for batch_ndx, sample in enumerate(loader):
        print(sample)
        print(batch_ndx)
