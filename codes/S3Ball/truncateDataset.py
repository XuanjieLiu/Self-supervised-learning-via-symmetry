import os
from os import path
from distutils.dir_util import copy_tree
import random

from tqdm import tqdm

FROM = './Ball3DImg/32_32_0.2_20_3_init_points_subset_2048'
TO   = './Ball3DImg/32_32_0.2_20_3_init_points_subset_2048_first_512'
NEW_SIZE = 512

def main():
    os.makedirs(TO, exist_ok=True)
    to_take = random.sample(os.listdir(FROM), NEW_SIZE)
    for traj_name in tqdm(to_take):
        copy_tree(
            path.join(FROM, traj_name), 
            path.join(TO  , traj_name), 
        )
    print('ok')

main()
