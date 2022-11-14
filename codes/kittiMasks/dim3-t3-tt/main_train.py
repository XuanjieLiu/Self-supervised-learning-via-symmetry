from train_config import CONFIG
from codes.kittiMasks.S3Model.trainer_symmetry import Trainer, is_need_train

trainer = Trainer(CONFIG)
if is_need_train(CONFIG):
    trainer.train()
