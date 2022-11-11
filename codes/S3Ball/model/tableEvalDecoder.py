from os import path
import tkinter as tk
from typing import List
from dataclasses import dataclass
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except ImportError:
    pass    # not Windows

import numpy as np
import torch
from PIL import Image, ImageTk
from PIL.Image import Resampling

from shared import DEVICE, loadModel
from normal_rnn import Conv2dGruConv2d, LAST_CN_NUM, LAST_H, LAST_W

RADIUS = 6
TICK_INTERVAL = .5
IMG_SIZE = 300

@dataclass
class Group():
    dir_name: str
    display: str
    config: dict

class UI:
    def __init__(
        self, groups: List[Group], n_rand_inits, 
        experiment_path, pt_name, 
        exp_name, 
    ):
        self.win = tk.Tk()
        self.win.title('Eval decoder')

        self.groups = groups
        self.n_row = n_rand_inits
        self.n_col = len(groups)
        self.models = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        self.photoLabels: List[List[tk.Label]] = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        self.photos = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        self.max_latent_dim = 0
        for col_i, group in enumerate(groups):
            self.max_latent_dim = max(
                self.max_latent_dim, 
                group.config['latent_code_num'], 
            )
            groupLabel = tk.Label(
                self.win, text=group.display, 
            )
            groupLabel.grid(
                row = 1, 
                column = self.max_latent_dim + col_i, 
                # padx=5, pady=5, 
            )
            for row_i in range(self.n_row):
                try:
                    model = loadModel(
                        Conv2dGruConv2d, path.join(
                            experiment_path, group.dir_name, 
                            f'rand_init_{row_i}', 
                            pt_name, 
                        ), group.config, 
                    )
                except FileNotFoundError:
                    print('warn: checkpoint not found, skipping.')
                    model = None
                else:
                    model.eval()
                self.models[row_i][col_i] = model

                label = tk.Label(self.win)
                label.grid(
                    row = row_i + 2, 
                    column = self.max_latent_dim + col_i, 
                    padx=5, pady=5, 
                )
                self.photoLabels[row_i][col_i] = label

        self.z = torch.zeros((
            self.max_latent_dim, 
        ), dtype=torch.float, device=DEVICE)
        topLabel = tk.Label(self.win, text=exp_name)
        topLabel.grid(
            row=0, 
            column=self.max_latent_dim, 
            columnspan=self.n_col, 
            # padx=5, pady=5, 
        )

        self.sliders: List[tk.Scale] = []
        self.initSliders()

    def initSliders(self):
        for i in range(self.max_latent_dim):
            slider = tk.Scale(
                self.win,
                variable=tk.DoubleVar(value=self.z[i]),
                command=lambda value, index=i : (
                    self.onSliderUpdate(index, value)
                ),
                from_=+ RADIUS, to=- RADIUS,
                resolution=0.01, 
                tickinterval=TICK_INTERVAL if i == 0 else 0, 
                length=2000, width=100, sliderlength=100,
            )
            slider.grid(
                row=0, rowspan=self.n_row + 2, column=i, 
                padx=10, 
            )
            self.sliders.append(slider)

    def onSliderUpdate(self, index, value):
        value = float(value)
        self.z[index] = value
        # print(self.z)
        self.sliders[index].set(value)
        for row_i, vae_row in enumerate(self.models):
            for col_i, vae in enumerate(vae_row):
                if vae is None:
                    continue
                img = decode(vae, self.z)
                img = img.resize((
                    IMG_SIZE, IMG_SIZE, 
                ), resample=Resampling.NEAREST)
                photo = ImageTk.PhotoImage(img)
                self.photos[row_i][col_i] = photo
                self.photoLabels[row_i][col_i].config(image=photo)

def decode(model: Conv2dGruConv2d, z: torch.Tensor):
    recon: torch.Tensor = model.decoder(model.fc3(z).view(1, LAST_CN_NUM, LAST_H, LAST_W))
    arr = (
        recon[0, :, :, :].cpu().detach().clamp(0, 1)
        .permute(1, 2, 0) * 255
    ).round().numpy().astype(np.uint8)
    img = Image.fromarray(arr, 'RGB')
    return img
