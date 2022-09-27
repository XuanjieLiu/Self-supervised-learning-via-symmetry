import os
from functools import lru_cache
from io import StringIO

import torch
import torch.utils.data
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pyperclip

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


def calc_group_mse(expGroup):
    dataset = Dataset(expGroup.z_coords_map_path)
    X = dataset.X
    std = dataset.Y.std(dim=0)
    Y = dataset.Y / std
    err = getErr(X, Y)
    x_mse = err[:, 0].square().mean().item()
    y_mse = err[:, 1].square().mean().item()
    z_mse = err[:, 2].square().mean().item()

    xyz_mse = (x_mse + z_mse + y_mse) / 3
    return x_mse, y_mse, z_mse, xyz_mse

def main():
    cols = [[] for _ in range(4)]
    for row_i, expGroup in enumerate(QUANT_EXP_GROUPS):
        for col_i, result in enumerate(calc_group_mse(expGroup)):
            cols[col_i].append(result)
    for col in cols:
        _min = min(col)
        for row_i, result in enumerate(col):
            s = format(result, '.2f')
            if result == _min:
                s = r'\bm{%s}' % s
            s = '$%s$' % s
            col[row_i] = s
    
    sIO = StringIO()
    print(r'\begin{center}', file=sIO)
    print(r'\begin{tabular}{l|cccc}', file=sIO)

    print('Method ', end='', file=sIO)
    for dim in 'xyz':
        # print(
        #     '& MSE on $d_%s / \mathrm{std}(d_%s)$ \downarrow ' 
        #     % (dim, dim), end='', file=sIO, 
        # )
        print('& $', dim, '$ axis MSE $\downarrow$ ', sep='', end='', file=sIO)
    print('& MSE $\downarrow$ ', file=sIO)
    print(r'\\ \hline', file=sIO)

    for row_i, expGroup in enumerate(QUANT_EXP_GROUPS):
        print(expGroup.display_name, end='', file=sIO)
        for col in cols:
            print('&', col[row_i], '', end='', file=sIO)
        if row_i < 2:
            print(r'\\ ', file=sIO)
    
    print(r'\end{tabular}', file=sIO)
    print(r'\end{center}', file=sIO)

    sIO.seek(0)
    s = sIO.read()
    print()
    print()
    print(s)
    print()
    print()
    pyperclip.copy(s)
    print('Copied to clipboard.')

if __name__ == '__main__':
    main()
