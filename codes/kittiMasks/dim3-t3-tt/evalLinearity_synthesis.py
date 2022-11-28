
# Encode the bouncing ball video test set. 
# i.e. generate the linear projection training set. 

from typing import List

import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure
from tqdm import tqdm

from codes.kittiMasks.S3Model.normal_rnn import (
    Conv2dGruConv2d, 
    LAST_CN_NUM, LAST_H, LAST_W, IMG_CHANNEL, 
)
from train_config import CONFIG
from codes.S3Ball.shared import *
from evalLinearity_shared import *


Z_RADIUS = 2
N_LATENT_DIM = 5

EXTENT = [-Z_RADIUS, Z_RADIUS, -Z_RADIUS, Z_RADIUS]

RAINBOW_EXP_GROUPS: List[ExpGroup] = [
    ours, 
    ablat, 
    # beta_vae, 
    # ae_1, ae_2, 
]

plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'font.size': 14,
    })



def hideTicks(ax: Axes):
    ax.tick_params(
        axis='both', which='both', 
        left=False, 
        bottom=False, 
        labelleft=False, 
        labelbottom=False, 
    )

def plotNDims(n=3):
    FIGSIZE = (11, 3)
    NECK_LINE_X = .02
    WIDTH_RATIOS = [.33, .02, .3, .02, .3]
    LEN_Z_LADDER = 5
    assert LEN_Z_LADDER % 2 == 1  # to show z=0
    Z_LADDER = torch.linspace(-Z_RADIUS, Z_RADIUS, LEN_Z_LADDER)

    fig = plt.figure(constrained_layout=True, figsize=FIGSIZE)
    N_SUBFIGS = len(VISUAL_EXP_GROUPS) * 2 - 1
    subfigs: List[SubFigure] = fig.subfigures(
        1, N_SUBFIGS, width_ratios=WIDTH_RATIOS, 
    )
    for expGroup_i, subfig in enumerate(subfigs[0::2]):
        expGroup = VISUAL_EXP_GROUPS[expGroup_i]
        model = loadModel(expGroup)
        axeses: List[List[Axes]] = subfig.subplots(
            n, LEN_Z_LADDER, 
            sharex=True, sharey=True, 
        )
        for row_i, axes in tqdm(enumerate(axeses), expGroup.display_name):
            for col_i, ax in enumerate(axes):
                z_val = Z_LADDER[col_i]
                z = torch.zeros((expGroup.n_latent_dims, ))
                z[row_i] = z_val
                img = synth(model, z).cpu()
                img = img.repeat(1, 1, 3)
                ax.imshow(img, extent=EXTENT)
                hideTicks(ax)
                if expGroup_i == 0 and col_i == 0:
                    ax.set_ylabel(
                        '$z_%d$' % (row_i + 1), 
                        rotation=0, 
                    )
                    ax.yaxis.set_label_coords(-.4, .3)
                if row_i == n - 1:
                    if col_i % 2 == 0:
                        ax.set_xlabel(
                            '$%.1f$' % z_val, 
                        )
        subfig.suptitle(
            '\\textbf{(%s) %s}' % (
                'abc'[expGroup_i], expGroup.display_name
            )
        )
    plt.show()

def synth(model: Conv2dGruConv2d, *args):
    if len(args) == 1:
        z = args[0]
    elif len(args) == 2:
        z_location, z_color = args
        z = torch.concat((z_location, z_color))
    else:
        raise TypeError('argument wrong')
    z = z.to(DEVICE)
    sample = model.decoder(model.fc3(z).view(
        1, LAST_CN_NUM, LAST_H, LAST_W, 
    )).detach()
    video: torch.Tensor = sample.data.view(
        1, IMG_CHANNEL, sample.size(2), sample.size(3), 
    )
    return video[0, :, :, :].permute(
        # before: color channel (rgb), width, height
        1, 2, 0, 
    )


plotNDims()