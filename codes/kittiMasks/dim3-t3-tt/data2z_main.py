from train_config import CONFIG
from codes.kittiMasks.S3Model.trainer_symmetry import Trainer, is_need_train

trainer = Trainer(CONFIG, is_train=False)
CHECK_POINT = 'our0.1416t3vae.pt'
if is_need_train(CONFIG):
    trainer.data2z(CHECK_POINT, 'data2z.txt')
