'''
util to zip training results.  
'''

import os
from os import path
import shutil

from tqdm import tqdm

DIRS = ['R2', 'T2', 'T1R2', 'T2R3', 'T3R2']
ONLY_LATEST = 'only_latest'

os.makedirs(ONLY_LATEST, exist_ok=True)
for dir_name in tqdm(DIRS):
    os.chdir(dir_name)
    for k in (0, 1, 4):
        for rand_i in range(6):
            r_dir = f'k={k}_rand_init_{rand_i}'
            dest = path.join('..', ONLY_LATEST, dir_name, r_dir)
            os.makedirs(dest, exist_ok=True)
            for fn in (
                'latest.pt', '../train_config.py', 
                'Train_record.txt', 'Eval_record.txt', 
            ):
                try:
                    shutil.copyfile(
                        path.join(r_dir, fn), 
                        path.join(dest,  fn), 
                    )
                except FileNotFoundError as e:
                    print(
                        'warning:', e, f'{dir_name = }', 
                        # f'{r_dir = }', f'{fn = }', 
                    )
    os.chdir('..')
print('taring...')
os.system(f'tar -czf {ONLY_LATEST}.tar.gz {ONLY_LATEST}')
