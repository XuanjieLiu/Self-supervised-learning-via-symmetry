import math
import sys
import os
import numpy as np
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../'))
import matplotlib.pyplot as plt
from typing import List


EVAL_RECORD_NAME = 'Eval_record.txt'
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 7,
})

BATCH_EXP = [
    {
        'dir': 'beta_vae',
        'name': 'beta-VAE'
    }, {
        'dir': 'noSymm_vae',
        'name': 'Ours w/o symmetry'
    }, {
        'dir': 't3_k1_ae',
        'name': 'T3 AE',
    }, {
        'dir': 't3_k1_vae',
        'name': 'T3 VAE',
    }, {
        'dir': 't3_r2_k1_ae',
        'name': 'T3 R2 AE',
    }, {
        'dir': 't3_r2_k1_vae',
        'name': 'T3 R2 VAE',
    },
]

DATA_SIZE_ORDER = [256, 512, 1024, 2048]
SYMM_ORDER = [0, 4, 16]
BATCH_ROOT = "./"
SUB_EXP_LIST = [str(i) for i in range(1, 31)]


class BatchExpResult:
    def __init__(self):
        self.path = None
        self.name = None
        self.primary_best_results = []
        self.secondary_best_results = []
        self.third_best_result = []
        self.best_idx = []


def find_results2best_primary(primary_idx: int, secondary_idx: int, third_idx:int, eval_result_path):
    """0-self_recon, 1-prior_recon, 2-self_recon_norm, 3-prior_recon_norm, 4-prior_z_norm, 5-xyz_mse"""
    with open(eval_result_path, 'r') as f:
        nums_lines = [[float(item.split(':')[-1]) for item in l.split('-')[-1].split(',')] for l in f.readlines()]
    primary_list = [line[primary_idx] for line in nums_lines]
    secondary_list = [line[secondary_idx] for line in nums_lines]
    third_list = [line[third_idx] for line in nums_lines]
    best_primary = min(primary_list)
    best_idx = primary_list.index(best_primary)
    secondary = secondary_list[best_idx]
    third = third_list[best_idx]
    return best_primary, secondary, third, best_idx


def stat_batch_result(batch_path_root, primary_idx, secondary_idx, third_idx):
    batch_result_list = []
    for exp in BATCH_EXP:
        br = BatchExpResult()
        br.name = exp['name']
        exp_path = os.path.join(batch_path_root, exp['dir'])
        sub_exp_list = list(filter(lambda sub_exp_name: sub_exp_name in SUB_EXP_LIST, os.listdir(exp_path)))
        if len(sub_exp_list) == 0:
            print(f"No sub_exp in {exp_path}")
            continue
        for sub_exp in sub_exp_list:
            eval_result_path = os.path.join(exp_path, sub_exp, EVAL_RECORD_NAME)
            max_primary, secondary, third, best_idx = find_results2best_primary(primary_idx, secondary_idx, third_idx, eval_result_path)
            br.primary_best_results.append(max_primary)
            br.secondary_best_results.append(secondary)
            br.third_best_result.append(third)
            br.best_idx.append(best_idx)
        batch_result_list.append(br)
    print('done')
    return batch_result_list


def stat_projmse_by_recon(batch_path_root):
    batch_result_list = []
    for exp in BATCH_EXP:
        br = BatchExpResult()
        br.name = exp['name']
        br.path = exp['dir']
        exp_path = os.path.join(batch_path_root, exp['dir'])
        # sub_exp_list = list(filter(lambda sub_exp_name: sub_exp_name in SUB_EXP_LIST, os.listdir(exp_path)))
        sub_exp_list = SUB_EXP_LIST
        print(sub_exp_list)
        if len(sub_exp_list) == 0:
            print(f"No sub_exp in {exp_path}")
            continue
        """primary: self-recon; secondary: pred-recon; third: proj_mse"""
        for sub_exp in sub_exp_list:
            """0-self_recon, 1-prior_recon, 2-self_recon_norm, 3-prior_recon_norm, 4-prior_z_norm, 5-xyz_mse"""
            eval_result_path = os.path.join(exp_path, sub_exp, EVAL_RECORD_NAME)
            if br.path == 'beta_vae':
                max_primary, secondary, third, best_idx = find_results2best_primary(2, 3, 5, eval_result_path)
                br.primary_best_results.append(max_primary)
                br.secondary_best_results.append(secondary)
            else:
                max_primary, secondary, third, best_idx = find_results2best_primary(3, 2, 5, eval_result_path)
                br.primary_best_results.append(secondary)
                br.secondary_best_results.append(max_primary)
            br.third_best_result.append(third)
            br.best_idx.append(best_idx)
        batch_result_list.append(br)
    print('done')
    return batch_result_list


def sort_batch_result_list_by_order_list(batch_result_list: List[BatchExpResult], datasize_order, symm_order) -> List[
    BatchExpResult]:
    sorted_list = []
    for ds in datasize_order:
        for sy in symm_order:
            select = list(filter(lambda br: br.symm == sy and br.datasize == ds, batch_result_list))[0]
            sorted_list.append(select)
    return sorted_list


def plot_fig_1(batch_result_list: List[BatchExpResult], ax):
    for exp in batch_result_list:
        x = exp.third_best_result
        y = exp.primary_best_results
        ax.scatter(x, y, label=exp.name)
    ax.legend()
    ax.set(ylabel='Loss on $\hat{\\textbf{x}}$', xlabel='Linear proj. loss')


def plot_loss(batch_result_list: List[BatchExpResult]):
    fig, axs = plt.subplots(1, 1, figsize=(5.5, 2))
    plot_fig_1(batch_result_list, axs)
    plt.show()


def calc_result_mean(batch_result_list: List[BatchExpResult]):
    result_list = []
    for exp in batch_result_list:
        result = {
            'name': exp.name,
            'self-recon mean': np.mean(exp.primary_best_results),
            'self-recon std': np.std(exp.primary_best_results),
            'pred-recon mean': np.mean(exp.secondary_best_results),
            'pred-recon std': np.std(exp.secondary_best_results),
            'proj.mse mean': np.mean(exp.third_best_result),
            'proj.mse std': np.std(exp.third_best_result),
        }
        result_list.append(result)
    return result_list



if __name__ == '__main__':
    batch_result_list = stat_projmse_by_recon(BATCH_ROOT)
    result_mean = calc_result_mean(batch_result_list)
    print(result_mean)
    plot_loss(batch_result_list)
    print('aaa')

