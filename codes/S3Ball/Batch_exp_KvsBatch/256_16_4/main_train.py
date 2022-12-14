import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../../'))
from codes.common_utils import create_results_path_if_not_exist
from train_config import CONFIG
from codes.S3Ball.Batch_exp_KvsBatch.trainer_symmetry import BallTrainer, is_need_train

EXP_NUM = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
FILE_DIR = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/')

for exp in EXP_NUM:
    sub_path = f'{FILE_DIR}/{exp}'
    create_results_path_if_not_exist(sub_path)
    os.chdir(sub_path)
    if is_need_train(CONFIG):
        trainer = BallTrainer(CONFIG)
        trainer.train()



