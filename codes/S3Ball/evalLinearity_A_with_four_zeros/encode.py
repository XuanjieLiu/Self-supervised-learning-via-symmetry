import torch

from ..Continue_ColorfulBall_Rnn256_45dimColor.normal_rnn import Conv2dGruConv2d, BATCH_SIZE
from ..Continue_ColorfulBall_Rnn256_45dimColor.train_config import CONFIG
from ..ball_data_loader import BallDataLoader
from shared import *

DATASET_SIZE = 1024

def main():
    dataLoader = BallDataLoader(CONFIG['eval_data_path'], True)
    for expGroup in EXP_GROUPS:
        model = Conv2dGruConv2d(CONFIG).to(DEVICE)
        model.load_state_dict(torch.load(
            expGroup.checkpoint_path, 
            map_location=DEVICE,
        ))
        model.eval()
        for batch in OneDataset(dataLoader):
            # batch: i_in_batch, color_channel, x, y
            z, mu, logvar = model.batch_seq_encode_to_z(batch)
            mu: torch.Tensor
            z_pos = mu[..., 0:3].detach()
            
main()
