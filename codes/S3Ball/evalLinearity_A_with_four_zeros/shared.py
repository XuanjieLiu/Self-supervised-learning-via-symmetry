from typing import List
import torch

class ExpGroup:
    def __init__(self) -> None:
        self.checkpoint_path = None
        self.display_name = None

EXP_GROUPS: List[ExpGroup] = []

eG = ExpGroup()
eG.checkpoint_path = './checkpoints/continue_symm0_VAE_cp150000_(AB).pt'
eG.display_name = 'VAE+RNN, no Symmetry'
EXP_GROUPS.append(eG)

eG = ExpGroup()
eG.checkpoint_path = './checkpoints/continue_symm4_VAE_1_cp140000.pt'
eG.display_name = 'VAE+RNN, with Symmetry'
EXP_GROUPS.append(eG)

eG = ExpGroup()
eG.checkpoint_path = './checkpoints/continue_symm4_AE_1_cp130000.pt'
eG.display_name = 'AE+RNN 1, with Symmetry'
EXP_GROUPS.append(eG)

eG = ExpGroup()
eG.checkpoint_path = './checkpoints/continue_symm4_AE_2_cp150000.pt'
eG.display_name = 'AE+RNN 2, with Symmetry'
EXP_GROUPS.append(eG)

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
CPU  = torch.device("cpu")
if HAS_CUDA:
    DEVICE = CUDA
    print('We have CUDA.')
else:
    DEVICE = CPU
    print("We DON'T have CUDA.")
