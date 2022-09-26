# Encode the bouncing ball video test set. 
# i.e. generate the linear projection training set. 

import torch
from tqdm import tqdm

from Continue_ColorfulBall_Rnn256_45dimColor.normal_rnn import Conv2dGruConv2d, BATCH_SIZE
from Continue_ColorfulBall_Rnn256_45dimColor.train_config import CONFIG
from ball_data_loader import BallDataLoader
from shared import *
from evalLinearity_A_with_four_zeros_shared import *

DATASET_SIZE = 1024

def main():
    dataLoader = BallDataLoader(
        './Ball3DImg/32_32_0.2_20_3_init_points_colorful_continue_evalset/', 
        True, 
    )
    for expGroup in tqdm(EXP_GROUPS):
        config = {
            **CONFIG, 
            'latent_code_num': expGroup.n_latent_dims, 
        }
        model = Conv2dGruConv2d(config).to(DEVICE)
        model.load_state_dict(torch.load(
            expGroup.checkpoint_path, 
            map_location=DEVICE,
        ))
        model.eval()
        with open(expGroup.z_coords_map_path, 'w') as f:
            for batch, trajectory in dataLoader.IterWithPosition(BATCH_SIZE):
                # batch:      i_in_batch, t, color_channel, x, y
                # trajectory: i_in_batch, t, coords_i
                z, mu, logvar = model.batch_seq_encode_to_z(batch)
                mu: torch.Tensor
                z_pos = mu[..., :3].detach()
                # z_pos: i_in_batch, t, i_in_z
                for i_in_batch in range(BATCH_SIZE):
                    for t in range(trajectory.shape[1]):
                        line = []
                        for z_i in z_pos[i_in_batch, t, :]:
                            line.append(repr(z_i.item()))
                        f.write(','.join(line))
                        f.write(' -> ')
                        line.clear()
                        for coord_i in trajectory[i_in_batch, t, :]:
                            line.append(repr(coord_i.item()))
                        f.write(','.join(line))
                        f.write('\n')

if __name__ == '__main__':
    main()
