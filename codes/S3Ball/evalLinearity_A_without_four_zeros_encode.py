# Encode the bouncing ball video test set. 
# i.e. generate the linear projection training set. 

import torch
from tqdm import tqdm

from ball_data_loader import BallDataLoader
from shared import *
from evalLinearity_shared import *


def data2z(dataset_path, exp_groups):
    dataLoader = BallDataLoader(
        dataset_path,
        True, 
    )
    for expGroup in tqdm(exp_groups):
        model = loadModel(expGroup)
        with open(expGroup.z_coords_map_path, 'w') as f:
            for batch, trajectory in dataLoader.IterWithPosition(BATCH_SIZE):
                batch = batch.to(DEVICE)
                trajectory = trajectory.to(DEVICE)
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
    data2z('./Ball3DImg/32_32_0.2_20_3_init_points_EvalSet/', QUANT_EXP_GROUPS)
