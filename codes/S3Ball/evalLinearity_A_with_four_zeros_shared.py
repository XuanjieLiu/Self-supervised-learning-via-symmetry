from typing import List

class ExpGroup:
    def __init__(self) -> None:
        self.checkpoint_path = None
        self.display_name = None
        self.z_coords_map_path: str = None

EXP_GROUPS: List[ExpGroup] = []

eG = ExpGroup()
eG.checkpoint_path = './evalLinearity_A_with_four_zeros/checkpoints/continue_symm0_VAE_cp150000_(AB).pt'
eG.display_name = 'VAE+RNN, no Symmetry'
eG.z_coords_map_path = './evalLinearity_A_with_four_zeros/z_coords_map/vae_aug_0.txt'
EXP_GROUPS.append(eG)

eG = ExpGroup()
eG.checkpoint_path = './evalLinearity_A_with_four_zeros/checkpoints/continue_symm4_VAE_1_cp140000.pt'
eG.display_name = 'VAE+RNN, with Symmetry'
eG.z_coords_map_path = './evalLinearity_A_with_four_zeros/z_coords_map/vae_aug_4.txt'
EXP_GROUPS.append(eG)

eG = ExpGroup()
eG.checkpoint_path = './evalLinearity_A_with_four_zeros/checkpoints/continue_symm4_AE_1_cp130000.pt'
eG.display_name = 'AE+RNN 1, with Symmetry'
eG.z_coords_map_path = './evalLinearity_A_with_four_zeros/z_coords_map/ae_aug_4_1.txt'
EXP_GROUPS.append(eG)

eG = ExpGroup()
eG.checkpoint_path = './evalLinearity_A_with_four_zeros/checkpoints/continue_symm4_AE_2_cp150000.pt'
eG.display_name = 'AE+RNN 2, with Symmetry'
eG.z_coords_map_path = './evalLinearity_A_with_four_zeros/z_coords_map/ae_aug_4_2.txt'
EXP_GROUPS.append(eG)
