import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../'))
from batch_exp_FindExpGroup import *
from evalLinearity_A_with_four_zeros_regress import calc_group_mse


def record_groups_mse(expGroups):
    for eg in expGroups:
        print(eg.display_name)
        x_mse, y_mse, z_mse, xyz_mse = calc_group_mse(eg)
        with open(eg.mse_result_path, 'w') as f:
            f.write(f'{x_mse},{y_mse},{z_mse},{xyz_mse}')


if __name__ == '__main__':
    eGs = gen_exp_groups(BATCH_ROOT)
    record_groups_mse(eGs)



