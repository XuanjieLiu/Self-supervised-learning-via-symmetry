import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../../'))
from codes.common_utils import create_results_path_if_not_exist
from train_config import CONFIG
from codes.kittiMasks.S3Model.trainer_symmetry import is_need_train, Trainer

EXP_NUM = [str(n) for n in list(range(1, 31))]
FILE_DIR = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/')

for exp in EXP_NUM:
    sub_path = f'{FILE_DIR}/{exp}'
    create_results_path_if_not_exist(sub_path)
    os.chdir(sub_path)
    if is_need_train(CONFIG):
        trainer = Trainer(CONFIG)
        trainer.train()



