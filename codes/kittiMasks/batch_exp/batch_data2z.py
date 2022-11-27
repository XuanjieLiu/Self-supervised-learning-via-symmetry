import sys
import os
from codes.kittiMasks.batch_exp.common_param import BATCH_EXP, SUB_EXP_LIST, CHECK_POINT_NUM_LIST
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../'))
from common_train_config import CONFIG
from codes.kittiMasks.S3Model.trainer_symmetry import Trainer


BATCH_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    trainer = Trainer(CONFIG, is_train=False)
    for exp in BATCH_EXP:
        exp_path = os.path.join(BATCH_ROOT, exp['dir'])
        os.chdir(exp_path)
        for sub_exp in SUB_EXP_LIST:
            sub_exp_path = os.path.join(exp_path, sub_exp)
            os.chdir(sub_exp_path)
            for num in CHECK_POINT_NUM_LIST:
                check_point_name = f'checkpoint_{num}.pt'
                output_name = f'data2z_{num}.txt'
                trainer.data2z(check_point_name, output_name)







