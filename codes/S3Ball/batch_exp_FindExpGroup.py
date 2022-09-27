import os
import sys
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../'))
from codes.S3Ball.evalLinearity_A_with_four_zeros_encode import data2z
from evalLinearity_shared import ExpGroup
from typing import List


CHECK_POINT_NUM_LIST = [i*10000 for i in range(4, 16)]
SUB_EXP_LIST = [str(i) for i in range(1, 11)]
DATASET_PATH = './Ball3DImg/32_32_0.2_20_3_init_points_EvalSet/'
BATCH_ROOT = "batch_exp/"
print(os.listdir("batch_exp/"))


def gen_exp_groups(batch_path_root):
    exp_group_list = []
    batch_path_list = os.listdir(batch_path_root)
    for batch_path in batch_path_list:
        exp_path = os.path.join(batch_path_root, batch_path)
        sub_exp_list = list(filter(lambda sub_exp_name: sub_exp_name in SUB_EXP_LIST, os.listdir(exp_path)))
        if len(sub_exp_list) == 0:
            print(f"No sub_exp in {batch_path}")
            continue
        for sub_exp in sub_exp_list:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            checkpoints_list = [f'checkpoint_{num}.pt' for num in CHECK_POINT_NUM_LIST]
            for checkpoint in checkpoints_list:
                add_checkpoint_to_exp_group_list(sub_exp_path, checkpoint, exp_group_list)
    return exp_group_list


def add_checkpoint_to_exp_group_list(checkpoint_parDir, checkpoint_name, exp_group_list: List[ExpGroup]):
    eG = ExpGroup()
    eG.checkpoint_path = os.path.join(checkpoint_parDir, checkpoint_name)
    eG.display_name = eG.checkpoint_path
    checkpoint_num = checkpoint_name.split('_')[-1].split('.')[0]
    eG.z_coords_map_path = os.path.join(checkpoint_parDir, f'z_coords_map_{checkpoint_num}.txt')
    eG.mse_result_path = os.path.join(checkpoint_parDir, f'mse_result_{checkpoint_num}.txt')
    eG.n_latent_dims = 3
    exp_group_list.append(eG)


if __name__ == '__main__':
    eGs = gen_exp_groups(BATCH_ROOT)
    data2z(DATASET_PATH, eGs)

