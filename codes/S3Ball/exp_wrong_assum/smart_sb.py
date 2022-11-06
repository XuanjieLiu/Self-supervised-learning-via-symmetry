#!/bin/python3

from datetime import datetime
from subprocess import Popen
import argparse

SBATCH_FILENAME = 'auto.sbatch'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_dir", type=str, nargs='?', 
        help="the dir, contains `train_config.py`, and to save checkpoints in.", 
    )
    args = parser.parse_args()

    config_dir = args.config_dir

    t = datetime.now().strftime('%Y_m%m_d%d@%H_%M_%S')

    with open('template.sbatch', 'r') as fin:
        with open(SBATCH_FILENAME, 'w') as fout:
            for line in fin:
                fout.write(line.replace(
                    '{OUT_FILENAME}', f'{t}_%j_%x', 
                ).replace(
                    '{CONFIG_DIR}', config_dir, 
                ))
    
    with Popen(['sbatch', SBATCH_FILENAME]) as p:
        p.wait()

main()
