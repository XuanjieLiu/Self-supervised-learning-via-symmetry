import os
from os import path
import sys
from typing import List
from dataclasses import dataclass

import torch
from matplotlib import pyplot as plt

from shared import DEVICE
from normal_rnn import Conv2dGruConv2d
from linearity_metric import projectionMSE

temp_dir = path.abspath(path.join(
    path.dirname(__file__), '../..', 
))
sys.path.append(temp_dir)
from S3Ball.ball_data_loader import BallDataLoader
assert sys.path.pop(-1) == temp_dir


BATCH_SIZE = 256

@dataclass
class Group():
    dir_name: str
    display: str
    config: dict

def loadModel(filename: str, config):
    model = Conv2dGruConv2d(config).to(DEVICE)
    model.load_state_dict(torch.load(
        filename, map_location=DEVICE,
    ))
    model.eval()
    return model

def evalEncoder(
    groups: List[Group], n_rand_inits, 
    lock_epoch, 
    dataset_path, experiment_path, 
):
    dataLoader = BallDataLoader(
        dataset_path,
        True, 
    )
    videos = []
    trajs = []
    for video, traj in dataLoader.IterWithPosition(BATCH_SIZE):
        _shape = video.shape
        videos.append(video.view(
            _shape[0] * _shape[1], _shape[2], _shape[3], _shape[4], 
        ))
        _shape = traj.shape
        trajs.append(traj.view(
            _shape[0] * _shape[1], _shape[2], 
        ))
    image_set = torch.cat(videos, dim=0).to(DEVICE)
    traj_set  = torch.cat(trajs,  dim=0).to(DEVICE)
    X = range(len(groups))
    Y = [[] for _ in range(n_rand_inits)]
    for group in groups:
        print(group.display)
        for rand_init_i in range(n_rand_inits):
            print(f'{rand_init_i = }')
            model = loadModel(
                path.join(
                    experiment_path, group.dir_name, 
                    f'rand_init_{rand_init_i}', 
                    f'checkpoint_{lock_epoch}.pt', 
                ), group.config, 
            )
            with torch.no_grad():
                z, mu, logvar = model.batch_seq_encode_to_z(batch)
            z_pos = mu[..., :3]
            mse = projectionMSE(z_pos, traj_set)
            Y[rand_init_i].append(mse)
    for Y_i in Y:
        plt.plot(
            X, Y_i, linestyle='none', 
            markerfacecolor='none', markeredgecolor='k', 
            marker='o', markersize=10, 
        )
    plt.ylabel('MSE')
    plt.xlabel(group.variable_name)
    plt.xticks(X, [g.variable_value for g in groups])
    plt.suptitle('Linear projection MSE (↓)')
    plt.savefig(path.join(experiment_path, 'auto_eval_encoder.pdf'))
    plt.show()
