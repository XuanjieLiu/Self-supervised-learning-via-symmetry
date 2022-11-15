import sys
import os

sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../'))
import matplotlib.pyplot as plt
from typing import List

plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'font.size': 7,
    })


DATA_SIZE_ORDER = [256, 512, 1024, 2048]
SYMM_ORDER = [0, 4, 16]
BATCH_ROOT = "./"
SUB_EXP_LIST = [str(i) for i in range(1, 11)]

class BatchExpResult:
    def __init__(self):
        self.path = None
        self.name = None
        self.primary_best_results = []
        self.secondary_best_results = []


def find_best_result_in_sub_exp(sub_exp_path):
    min_total_mse = 1000
    best_cp_num = 0
    best_x_mse = 0
    best_y_mse = 0
    best_z_mse = 0
    for cp_num in CHECK_POINT_NUM_LIST:
        mse_result_path = os.path.join(sub_exp_path, f'mse_result_{cp_num}.txt')
        with open(mse_result_path, 'r') as f:
            """0-x, 1-y, 2-z, 3-total"""
            result = [float(mse) for mse in f.readlines()[0].split(',')]
            if result[3] < min_total_mse:
                min_total_mse = result[3]
                best_cp_num = cp_num
                best_x_mse = result[0]
                best_y_mse = result[1]
                best_z_mse = result[2]
    return (
        min_total_mse, best_x_mse, best_y_mse, best_z_mse, best_cp_num
    )


def find_eval_result_by_iternum(sub_exp_path, iternum: int):
    split_result = lambda result: float(result.split(':')[-1])
    eval_result_path = os.path.join(sub_exp_path, 'Eval_record.txt')
    with open(eval_result_path, 'r') as f:
        s_lines = [l.split('-') for l in f.readlines()]
        result_line = list(filter(lambda x: int(x[0]) == iternum, s_lines))[-1]
        """0-self_recon, 1-prior_recon, 2-self_recon_norm, 3-prior_recon_norm, 4-prior_z_norm, 5-xyz_mse"""
        results = result_line[-1].split(',')
        return (
            split_result(results[0]),
            split_result(results[1]),
            split_result(results[4])
        )


def stat_batch_result(batch_path_root):
    batch_result_list = []
    batch_name_list = os.listdir(batch_path_root)
    for batch_name in batch_name_list:
        br = BatchExpResult()
        br.name = batch_name
        name_split = batch_name.split('_')
        br.datasize = int(name_split[2])
        br.symm = int(name_split[-1])
        exp_path = os.path.join(batch_path_root, batch_name)
        sub_exp_list = list(filter(lambda sub_exp_name: sub_exp_name in SUB_EXP_LIST, os.listdir(exp_path)))
        if len(sub_exp_list) == 0:
            print(f"No sub_exp in {batch_name}")
            continue
        for sub_exp in sub_exp_list:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            min_total_mse, best_x_mse, best_y_mse, best_z_mse, best_cp_num = find_best_result_in_sub_exp(sub_exp_path)
            br.best_mse_list.append(min_total_mse)
            br.best_cp_list.append(best_cp_num)
            l_self_recon, l_prior_recon, l_prior_z_norm = find_eval_result_by_iternum(sub_exp_path, best_cp_num)
            br.best_self_recon_list.append(l_self_recon)
            br.best_prior_recon_list.append(l_prior_recon)
            br.best_prior_z_norm_list.append(l_prior_z_norm)
        batch_result_list.append(br)
    print('done')
    return batch_result_list


def plot_box_rnn_recon(batch_result_list: List[BatchExpResult]):
    fig = plt.figure()  # 创建画布
    ax = plt.subplot()  # 创建作图区域
    # 缺口表示50%分位点的置信区间，缺口太大表示分布太分散了
    # ax.boxplot([range(5), range(10), range(20)], notch=True)
    ax.boxplot([br.best_prior_recon_list for br in batch_result_list], showfliers=False)
    # 修改x轴下标
    # ax.set_xticks()
    ax.set_xticklabels([f'{br.datasize}-{br.symm}' for br in batch_result_list])
    # 显示y坐标轴的底线
    plt.grid(axis='y')
    plt.show()


def sub_boxplot_by_datasize(batch_result_list: List[BatchExpResult]):
    fig, axs = plt.subplots(1, 4, figsize=(5.5, 2), sharey='all')

    def sup_plot():
        for i in range(4):
            brs = sort_batch_result_list_by_order_list(batch_result_list, [DATA_SIZE_ORDER[i]], SYMM_ORDER)
            axs[i].boxplot([br.best_mse_list for br in brs], showfliers=False)
            axs[i].set_title(f'{DATA_SIZE_ORDER[i]} Samples')
            axs[i].set_xticklabels([f'{br.symm}' for br in brs])
    sup_plot()

    for ax in axs.flat:
        ax.set(ylabel='Linear proj. loss', xlabel='K')
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


def sub_pointplot_by_datasize(batch_result_list: List[BatchExpResult]):
    colors_order = ['blue', 'gold', 'chocolate', 'red']
    fig, axs = plt.subplots(1, 4, figsize=(5.5, 2), sharey='all', sharex='all')
    sort_brl = sort_batch_result_list_by_order_list(batch_result_list, DATA_SIZE_ORDER, SYMM_ORDER)
    sub_exp_num = len(sort_brl[0].best_mse_list)
    scatter_handler = None
    def sup_plot():
        nonlocal scatter_handler
        for i in range(len(DATA_SIZE_ORDER)):
            # colors = [colors_order[j] for j in range(len(SYMM_ORDER)) for i in range(sub_exp_num)]
            for j in range(len(SYMM_ORDER)):
                opt_br = list(filter(lambda b: b.symm == SYMM_ORDER[j] and b.datasize == DATA_SIZE_ORDER[i], sort_brl))[0]
                x = [n for n in opt_br.best_mse_list]
                y = [n for n in opt_br.best_prior_recon_list]
                scatter_handler = axs[i].scatter(x, y, c=colors_order[j], label=f"$K={SYMM_ORDER[j]}$", s=2)
            axs[i].set_title(f'{DATA_SIZE_ORDER[i]} samples', pad=10)
    sup_plot()
    axs[-1].legend()
    for ax in axs.flat:
        ax.set(ylabel='Loss on $\hat{\\textbf{x}}$', xlabel='Linear proj. loss')
    for ax in axs.flat:
        ax.label_outer()

    plt.show()



def sort_batch_result_list_by_order_list(batch_result_list: List[BatchExpResult], datasize_order, symm_order) -> List[BatchExpResult]:
    sorted_list = []
    for ds in datasize_order:
        for sy in symm_order:
            select = list(filter(lambda br: br.symm == sy and br.datasize == ds, batch_result_list))[0]
            sorted_list.append(select)
    return sorted_list


if __name__ == '__main__':
    batch_result_list = stat_batch_result(BATCH_ROOT)
    sorted_brl = sort_batch_result_list_by_order_list(batch_result_list, DATA_SIZE_ORDER, SYMM_ORDER)
    sub_boxplot_by_datasize(sorted_brl)
    sub_pointplot_by_datasize(sorted_brl)
