import sys
import os
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../../../'))
from common_train_config import CONFIG
from codes.kittiMasks.S3Model.trainer_symmetry import Trainer

BATCH_EXP = [
    {
        'dir': 'beta_vae',
        'name': 'beta-VAE'
    }, {
        'dir': 'noSymm_vae',
        'name': 'Ours w/o symmetry'
    }, {
        'dir': 't3_k1_ae',
        'name': 'T3 AE',
    }, {
        'dir': 't3_k1_vae',
        'name': 'T3 VAE',
    }, {
        'dir': 't3_r2_k1_ae',
        'name': 'T3 R2 AE',
    }, {
        'dir': 't3_r2_k1_vae',
        'name': 'T3 R2 VAE',
    },
]

CHECK_POINT_NUM_LIST = [i*2000 for i in range(1, 11)]
SUB_EXP_LIST = [str(i) for i in range(1, 31)]

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







