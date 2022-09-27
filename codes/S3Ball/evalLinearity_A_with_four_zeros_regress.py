from functools import lru_cache

import torch
import torch.utils.data
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from shared import *
from evalLinearity_shared import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, n_loops=1) -> None:
        super().__init__()
        self.n_loops = n_loops
        X = []
        Y = []
        with open(file_path, 'r') as f:
            for line in f:
                z, coords = self.parseFileLine(line)
                X.append(torch.Tensor(z))
                Y.append(torch.Tensor(coords))
        self.X = torch.stack(X)
        self.Y = torch.stack(Y)
    
    def parseStrCoords(self, s: str):
        x, y, z = s.split(',')
        return float(x), float(y), float(z)

    def parseFileLine(self, line: str):
        z, coords = line.strip().split(' -> ')
        return self.parseStrCoords(z), self.parseStrCoords(coords)
    
    @lru_cache(1)
    def trueLen(self):
        return self.X.shape[0]
    
    @lru_cache(1)
    def __len__(self):
        return self.trueLen() * self.n_loops
    
    def __getitem__(self, index):
        return (
            self.X[index % self.trueLen()], 
            self.Y[index % self.trueLen()], 
        )


def PersistentLoader(dataset, batch_size):
    while True:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size, shuffle=True, 
            num_workers=1, 
        )
        for batch in loader:
            if batch.shape[0] != batch_size:
                break
            yield batch


def getErr(X, Y) -> torch.Tensor:
    regression = LinearRegression().fit(X, Y)
    return Y - regression.predict(X)


def getXZ(data: torch.Tensor):
    return torch.cat((
        data[:, 0].unsqueeze(1), 
        data[:, 2].unsqueeze(1), 
    ), dim=1)


def calc_group_mse(expGroup):
    dataset = Dataset(expGroup.z_coords_map_path)
    dataset.X[:, 0]
    xz_err = getErr(
        getXZ(dataset.X),
        getXZ(dataset.Y),
    )
    x_mse = xz_err[:, 0].square().mean().item()
    z_mse = xz_err[:, 1].square().mean().item()
    xz_mse = xz_err.square().mean().item()
    assert abs((x_mse + z_mse) * .5 - xz_mse) < 1e-5

    y_err = getErr(
        dataset.X[:, 1].unsqueeze(1),
        dataset.Y[:, 1].unsqueeze(1),
    )
    y_mse = y_err.square().mean().item()
    xyz_mse = (x_mse + z_mse + y_mse) / 3
    return x_mse, y_mse, z_mse, xyz_mse


if __name__ == '__main__':
    for expGroup in QUANT_EXP_GROUPS:
        print(expGroup.display_name)
        x_mse, y_mse, z_mse, xyz_mse = calc_group_mse(expGroup)
        print(  'x_mse', format(  x_mse, '.2f'))
        print(  'y_mse', format(  y_mse, '.2f'))
        print(  'z_mse', format(  z_mse, '.2f'))
        print('xyz_mse', format(xyz_mse, '.2f'))
