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

trainer = BallTrainer(CONFIG)
if is_need_train(CONFIG):
    trainer.train()
