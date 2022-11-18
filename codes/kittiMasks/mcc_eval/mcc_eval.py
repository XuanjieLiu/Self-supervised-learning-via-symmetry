# Under construction

import os
from os import path

from tqdm import tqdm

def main():
    os.chdir('Rnn256_DataSize_2048_symm_4_4')
    dirs = [x for x in os.listdir() if path.isdir(x)]
    for dir_name in tqdm(dirs):
        open(path.join(dir_name, None))

main()
