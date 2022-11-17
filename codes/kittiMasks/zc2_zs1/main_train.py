from train_config import CONFIG
from trainer_symmetry_SCSplit import Trainer, is_need_train

trainer = Trainer(CONFIG)
if is_need_train(CONFIG):
    trainer.train()
