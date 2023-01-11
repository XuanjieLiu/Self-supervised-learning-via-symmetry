#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=targz
#SBATCH --output=targz.out
#SBATCH --error=targz.err

cd ..
tar -vczf Dans_exp_singlColor.tar.gz Dans_exp_singlColor/
