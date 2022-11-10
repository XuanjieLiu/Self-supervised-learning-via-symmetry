import os
from os import path
import shutil

from tqdm import tqdm

DIRS = ['R2', 'T2', 'T1R2', 'T2R3', 'T3R2']
ONLY_LATEST = 'only_latest'

os.makedirs(ONLY_LATEST, exist_ok=True)
for dir_name in tqdm(DIRS):
    os.chdir(dir_name)
    for rand_i in range(6):
        r_dir = f'rand_init_{rand_i}'
        dest = path.join('..', ONLY_LATEST, r_dir)
        os.makedirs(dest, exist_ok=True)
        for fn in (
            'latest.pt', 
            'Train_record.txt', 'Eval_record.txt', 
        ):
            shutil.copyfile(fn, path.join(dest, fn))
    os.chdir('..')
print('taring...')
os.system(f'tar -czf {ONLY_LATEST}.tar.gz {ONLY_LATEST}')
