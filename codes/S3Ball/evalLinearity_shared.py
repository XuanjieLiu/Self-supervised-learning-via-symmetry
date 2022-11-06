from __future__ import annotations

from typing import List
from functools import lru_cache

from Continue_ColorfulBall_Rnn256_45dimColor.normal_rnn import Conv2dGruConv2d, BATCH_SIZE
from Continue_ColorfulBall_Rnn256_45dimColor.train_config import CONFIG
from shared import *

@lru_cache(16)
def loadModel(expGroup: ExpGroup):
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
    return model

class ExpGroup:
    def __init__(self) -> None:
        self.checkpoint_path = None
        self.display_name = None
        self.z_coords_map_path: str = None
        self.n_latent_dims = None
        self.mse_result_path = None


eG = ExpGroup()
eG.checkpoint_path = './evalLinearity_A_with_four_zeros/checkpoints/continue_symm0_VAE_cp150000_(AB).pt'
eG.display_name = 'Ours w/o Symmetry'
eG.z_coords_map_path = './evalLinearity_A_with_four_zeros/z_coords_map/vae_aug_0.txt'
eG.n_latent_dims = 5
ablat = eG

eG = ExpGroup()
eG.checkpoint_path = './evalLinearity_A_with_four_zeros/checkpoints/continue_symm4_VAE_1_cp140000.pt'
eG.display_name = 'SPS (Ours)'
eG.z_coords_map_path = './evalLinearity_A_with_four_zeros/z_coords_map/vae_aug_4.txt'
eG.n_latent_dims = 5
ours = eG

eG = ExpGroup()
eG.checkpoint_path = './evalLinearity_A_with_four_zeros/checkpoints/beta_vae_no_color_checkpoint_110000.pt'
eG.display_name = '$\\beta$-VAE (Baseline)'
eG.z_coords_map_path = './evalLinearity_A_with_four_zeros/z_coords_map/beta_vae.txt'
eG.n_latent_dims = 3
beta_vae = eG

eG = ExpGroup()
eG.checkpoint_path = './evalLinearity_A_with_four_zeros/checkpoints/beta_vae_checkpoint_correct.pt'
eG.display_name = '$\\beta$-VAE (Baseline)'
eG.z_coords_map_path = './evalLinearity_A_with_four_zeros/z_coords_map/beta_vae_color.txt'
eG.n_latent_dims = 5
beta_vae_color = eG

eG = ExpGroup()
eG.checkpoint_path = './evalLinearity_A_with_four_zeros/checkpoints/continue_symm4_AE_1_cp130000.pt'
eG.display_name = 'SPS-AE'
eG.z_coords_map_path = './evalLinearity_A_with_four_zeros/z_coords_map/ae_aug_4_1.txt'
eG.n_latent_dims = 5
ae_1 = eG

eG = ExpGroup()
eG.checkpoint_path = './evalLinearity_A_with_four_zeros/checkpoints/continue_symm4_AE_2_cp150000.pt'
eG.display_name = 'AE+RNN 2, with Symmetry'
eG.z_coords_map_path = './evalLinearity_A_with_four_zeros/z_coords_map/ae_aug_4_2.txt'
eG.n_latent_dims = 5
ae_2 = eG

if True:
    # normal
    QUANT_EXP_GROUPS: List[ExpGroup] = [
        beta_vae, ablat, ours, 
    ]
else:
    # vae vs ae
    ours.display_name = 'SPS-VAE'
    QUANT_EXP_GROUPS: List[ExpGroup] = [
        ae_1, ours, 
    ]

VISUAL_EXP_GROUPS: List[ExpGroup] = [
    beta_vae, ablat, ours, 
]
