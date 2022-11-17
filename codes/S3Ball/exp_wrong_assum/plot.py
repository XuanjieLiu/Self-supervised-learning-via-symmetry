from __future__ import annotations

from os import path
import sys
from importlib import reload
from typing import List
from dataclasses import dataclass
import pickle

import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


temp_dir = path.abspath(path.join(
    path.dirname(__file__), '../model', 
))
sys.path.append(temp_dir)
from shared import DEVICE, loadModel
from normal_rnn import Conv2dGruConv2d
from evalEncoder import Group, evalEncoder
from linearity_metric import projectionMSE, projectionMSELockHeight
assert sys.path.pop(-1) == temp_dir

temp_dir = path.abspath(path.join(
    path.dirname(__file__), '../..', 
))
sys.path.append(temp_dir)
from S3Ball.ball_data_loader import BallDataLoader
import S3Ball.rc_params as rc_params
assert sys.path.pop(-1) == temp_dir

rc_params.init(font_size=12)
plt.rc('text.latex', preamble=r'\usepackage{amssymb}')

BATCH_SIZE = 256
@dataclass
class Group():
    dir_name: str
    display: str
    config: dict
class USING_METRIC:
    name = ''
    method = projectionMSE
    suptitle = ''
group_names = ['T2', 'R2', 'T1R2', 'T3R2', 'T2R3']
groups: List[Group] = []
n_groups = len(group_names)
n_rand_inits = 6
Ks = (0, 1, 4)
n_Ks = len(Ks)
pt_name = 'latest.pt'
dataset_path = '../Ball3DImg/32_32_0.2_20_3_init_points_EvalSet/'
experiment_path = '.'

def fillConfigs():
    for group_name in group_names:
        temp_dir = path.abspath(group_name)
        sys.path.append(temp_dir)
        import train_config
        train_config = reload(train_config)
        assert sys.path.pop(-1) == temp_dir

        config = train_config.CONFIG
        for a in 'tr':
            print(a, ':', config[a + '_batch_multiple'], ' times with n_dims =', config[a + '_n_dims'])
        # input('Please verify. Enter...')
        print()
        g = Group(group_name, {
            'T2': '$$(\mathbb{R}^2, +)$$', 
            'R2': '$$\mathrm{SO}(2)$$', 
            'T1R2': '$$(\mathbb{R}^1, +) \\times \mathrm{SO}(2)$$', 
            'T3R2': '$$(\mathbb{R}^3, +) \\times \mathrm{SO}(2)$$', 
            'T2R3': '$$(\mathbb{R}^2, +) \\times \mathrm{SO}(3)$$', 
        }[group_name], config)
        groups.append(g)

def getData():
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
    print('dataset ready.')
    # Y = np.zeros((n_groups, n_Ks, n_rand_inits))
    Y = [[] for _ in range(n_groups * n_Ks)]
    for i_group, group in enumerate(tqdm(groups, 'encoding images')):
        print()
        print(group.display)
        for i_k, k in enumerate(Ks):
            print(f' {k = }')
            groupY = Y[i_group * n_Ks + i_k]
            for rand_init_i in range(n_rand_inits):
                print(f'  {rand_init_i = }')
                try:
                    model = loadModel(
                        Conv2dGruConv2d, path.join(
                            experiment_path, group.dir_name, 
                            f'k={k}_rand_init_{rand_init_i}', 
                            pt_name, 
                        ), group.config, 
                    )
                except FileNotFoundError:
                    print('   warn: checkpoint not found, skipping.')
                    groupY.append(None)
                    raise Exception
                    # Temporary fix
                    # groupY[-1] = 0
                    # from time import time
                    # if time() > 1668575354.2033336 + 60 * 60 * 5:
                    #     raise Exception
                    continue
                with torch.no_grad():
                    z, mu, logvar = model.batch_encode_to_z(
                        image_set, 
                    )
                z_pos = mu[..., :3]
                mse = projectionMSE(z_pos, traj_set)
                groupY.append(mse)
    return Y

def plot(Y):
    fig = plt.figure(figsize=(9, 3))
    axes = fig.subplots(1, n_groups, sharey=True)
    X = [*range(1, n_Ks + 1)]
    for col_i, ax in enumerate(axes):
        Ys = (Y[col_i], Y[col_i + n_groups],  Y[col_i + 2 * n_groups])
        # for x, y in zip(X, Ys):
        #     ax.plot(
        #         [x] * n_rand_inits, y, linestyle='none', 
        #         markerfacecolor='none', 
        #         markeredgecolor=(.6, ) * 3, 
        #         marker='o', markersize=10, 
        #     )
        ax.boxplot(Ys)
        ax.set_xlabel('$$K$$')
        ax.set_xticks(X)
        ax.set_xticklabels(Ks)
        ax.set_xlim(.8, 0.2 + n_Ks)
        ax.set_title(groups[col_i].display)
        ax.axhline(0, c='b')
    axes[0].set_ylabel('Linear proj. loss')
    # plt.suptitle('Linear projection MSE (↓)')
    plt.tight_layout()
    # plt.savefig(path.join(experiment_path, 'auto_eval_encoder.pdf'))
    plt.show()

def main():
    fillConfigs()
    SAVE_FILE = 'plot_cache.pickle'
    try:
        with open(SAVE_FILE, 'rb') as f:
            if input('Cache found. Use cache? y/n >') == 'y':
                Y = pickle.load(f)
            else:
                Y = None
    except FileNotFoundError:
        Y = None
    if Y is None:
        print('Getting data...')
        Y = getData()
        print('Caching...')
        with open(SAVE_FILE, 'wb') as f:
            pickle.dump(Y, f)
    print('plotting...')
    plot(Y)

main()
