import os
from os import path
import sys

try:
    # this branch for static analysis
    from ..model.trainer_symmetry import BallTrainer, is_need_train
except ImportError:
    # this branch for runtime
    temp_dir = path.abspath(path.join(
        path.dirname(__file__), '../model', 
    ))
    sys.path.append(temp_dir)
    from trainer_symmetry import BallTrainer, is_need_train
    assert sys.path.pop(-1) == temp_dir
else:
    raise RuntimeError('error egmw5n4i3p7regt')

try:
    # this branch for runtime
    from train_config import CONFIG
except ImportError:
    # this branch for static analysis
    from ..model.train_config import CONFIG
    if 1 == 1:
        raise RuntimeError('error o487wryhalegt')

# for K in (0, 1, 4):
for K in (0, 1):
    for rand_init_i in range(6):
        dir_name = f'k={K}_rand_init_{rand_init_i}'
        os.makedirs(dir_name, exist_ok=True)
        config = CONFIG.copy()
        config['train_result_path'] = path.join(
            dir_name, 'TrainingResults', 
        )
        config['train_record_path'] = path.join(
            dir_name, 'Train_record.txt', 
        )
        config['eval_record_path'] = path.join(
            dir_name, 'Eval_record.txt', 
        )
        config['model_path'] = path.join(
            dir_name, 'latest.pt', 
        )
        config['checkpoint_path'] = path.join(
            dir_name, 'checkpoint_%d.pt', 
        )
        for a in 'tr':
            for b in ('', 'recon_'): 
                field = f'{a}_{b}batch_multiple'
                if CONFIG[field] != 0:
                    assert CONFIG[field] == 4
                    # Doing the rest of the experiments. 
                    # Remove the assert if you are starting from scratch. 
                    config[field] = K

        trainer = BallTrainer(config)
        if is_need_train(config):
            trainer.train()
