# You don't need this script. 
# This is a temp util script for moving prev exps to K=4 folders. 

import os

DIRS = ['R2', 'T2', 'T1R2', 'T2R3', 'T3R2']

K = 4

for d in DIRS:
    os.chdir(d)
    for i in range(6):
        os.rename(f'rand_init_{i}', f'k={K}_rand_init_{i}')
    os.chdir('..')
