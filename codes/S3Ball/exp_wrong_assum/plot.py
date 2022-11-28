from __future__ import annotations

import os
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

rc_params.init()
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
            'T1R2': '$$(\mathbb{R}^1, +) $$\n$$ \\times $$\n$$ \mathrm{SO}(2)$$', 
            'T3R2': '$$(\mathbb{R}^3, +) $$\n$$ \\times $$\n$$ \mathrm{SO}(2)$$', 
            'T2R3': '$$(\mathbb{R}^2, +) $$\n$$ \\times $$\n$$ \mathrm{SO}(3)$$', 
        }[group_name]
            # .replace('$$\n$$ ', '')
        , config)
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
    Y = np.zeros((n_groups, n_Ks, n_rand_inits))
    for i_group, group in enumerate(tqdm(groups, 'encoding images')):
        print()
        print(group.display)
        for i_k, k in enumerate(Ks):
            print(f' {k = }')
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
                    Y[i_group, i_k, rand_init_i] = -1
                    from time import time
                    if time() > 1668660579.0168521 + 60 * 60 * 2:
                        raise Exception
                    continue
                with torch.no_grad():
                    z, mu, logvar = model.batch_encode_to_z(
                        image_set, 
                    )
                z_pos = mu[..., :3]
                mse = projectionMSE(z_pos, traj_set)
                Y[i_group, i_k, rand_init_i] = mse
    return Y

def plotMultiPanels(Y, baseline):
    fig = plt.figure(figsize=(9, 3))
    axes = fig.subplots(1, n_groups, sharey=True)
    X = [*range(1, n_Ks + 1)]
    for col_i, ax in enumerate(axes):
        groupY = [Y[col_i, ik, :] for ik in range(n_Ks)]
        groupY[0] = baseline
        ax.boxplot(groupY)
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

def plotSinglePanel(Y, baseline, ours):
    plt.figure(figsize=(6, 3))
    X = [*range(1, n_groups + 3)]
    listY = [baseline]
    assert Ks[2] == 4   # I want k=4
    for ig in range(n_groups):
        listY.append(Y[ig, 2, :])
    listY.append(ours)
    plt.boxplot(listY)
    # plt.xlabel('Incorrect Group Assumption')
    plt.xticks(
        X, ['w/o \n Symmetry', *[
            g.display for g in groups
        ], '$$(\mathbb{R}^2, +) $$\n$$ \\times $$\n$$ \mathrm{SO}(2)$$\n(Ours)'], 
        # rotation=-90, 
    )
    # plt.xlim(.8, 0.2 + n_groups)
    plt.ylim(bottom=0)
    # plt.axhline(0, c='b')
    plt.ylabel('Linear proj. loss')
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

    baseline = np.zeros((n_groups, n_rand_inits))
    assert Ks[0] == 0
    for ig in range(n_groups):
        for ir in range(n_rand_inits):
            baseline[ig, ir] = Y[ig, 0, ir]
    baseline = baseline.flatten()

    ours = getOurs()

    print('plotting...')
    plotSinglePanel(Y, baseline, ours)

def getOurs():
    os.chdir('Rnn256_DataSize_2048_symm_4_4')
    dirs = [x for x in os.listdir() if path.isdir(x)]
    ours = []
    for dir_name in tqdm(dirs):
        with open(path.join(
            dir_name, 'mse_result_150000.txt', 
        ), 'r') as f:
            mses = f.read().strip().split(',')
            mses = [float(x) for x in mses]
            assert abs(sum(mses[:3]) / 3 - mses[3]) < .0000001
            ours.append(mses[3])
    os.chdir('..')
    return ours

main()
