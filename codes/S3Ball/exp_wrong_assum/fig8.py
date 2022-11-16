from __future__ import annotations

from os import path
import sys
from importlib import reload
from typing import List
from dataclasses import dataclass

import torch
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
assert sys.path.pop(-1) == temp_dir

@dataclass
class Group():
    dir_name: str
    display: str
    config: dict
group_names = ['T2', 'R2', 'T1R2', 'T3R2', 'T2R3']
groups: List[Group] = []
class USING_METRIC:
    name = ''
    method = projectionMSE
    suptitle = ''
BATCH_SIZE = 256

def main():
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
        g = Group(group_name, group_name, config)
        groups.append(g)

    n_rand_inits = 6

    pt_name = 'latest.pt'
    dataset_path = '../Ball3DImg/32_32_0.2_20_3_init_points_EvalSet/'
    experiment_path = '.'

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
    Y = [[] for _ in range(6 * 6)]
    for i_group, group in enumerate(tqdm(groups, 'encoding images')):
        print()
        print(group.display)
        for i_k, k in enumerate((0, 1, 4)):
            groupY = Y[i_group * 6 + i_k]
            for rand_init_i in range(n_rand_inits):
                # print(f'{rand_init_i = }')
                try:
                    model = loadModel(
                        Conv2dGruConv2d, path.join(
                            experiment_path, group.dir_name, 
                            f'k={k}_rand_init_{rand_init_i}', 
                            pt_name, 
                        ), group.config, 
                    )
                except FileNotFoundError:
                    print('warn: checkpoint not found, skipping.')
                    groupY.append(None)
                    continue
                with torch.no_grad():
                    z, mu, logvar = model.batch_encode_to_z(
                        image_set, 
                    )
                z_pos = mu[..., :3]
                mse = projectionMSE(z_pos, traj_set)
                groupY.append(mse)
    fig, axes = plt.subplots(1, 6, sharey=True)
    X = [1, 2, 3]
    for col_i, ax in enumerate(axes):
        Ys = (Y[col_i], Y[col_i + 6],  Y[col_i + 12])
        for x, y in zip(X, Ys):
            ax.plot(
                [x] * n_rand_inits, y, linestyle='none', 
                markerfacecolor='none', markeredgecolor='k', 
                marker='o', markersize=10, 
            )
        ax.boxplot(Ys)
        ax.set_xticks(X)
        ax.set_xticklabels(['symm', 'no symm'])
        ax.set_xlim(.8, 3.2)
        ax.set_title(group.display)
    axes[0].set_ylabel('MSE')
    plt.suptitle('Linear projection MSE (↓)' 
        + USING_METRIC.suptitle
    )
    plt.tight_layout()
    plt.savefig(path.join(experiment_path, 'auto_eval_encoder.pdf'))
    plt.show()

main()
