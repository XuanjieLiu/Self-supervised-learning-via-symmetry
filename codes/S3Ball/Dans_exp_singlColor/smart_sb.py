#!/bin/python3

import os
from subprocess import Popen

TASKS = [
    'ae_noSymm', 
    'ae_symm4', 
    'vae_noSymm', 
    'vae_symm4', 
    'beta_vae', 
]

SBATCH_FILENAME = 'auto.sbatch'

def main():
    user = os.getenv('USER')

    for task in TASKS:
        with open('template.sbatch', 'r') as fin:
            with open(SBATCH_FILENAME, 'w') as fout:
                for line in fin:
                    fout.write(line.replace(
                        '{OUT_FILENAME}', f'%j_%x', 
                    ).replace(
                        '{JOB_NAME}', task, 
                    ).replace(
                        '{CMDS}', f'cd {task}; python main_train.py', 
                    ).replace(
                        '{USER}', user, 
                    ))
    
        with Popen(['sbatch', SBATCH_FILENAME]) as p:
            p.wait()

main()
