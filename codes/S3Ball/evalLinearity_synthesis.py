
# Encode the bouncing ball video test set. 
# i.e. generate the linear projection training set. 

from typing import List

import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure
from tqdm import tqdm

from Continue_ColorfulBall_Rnn256_45dimColor.normal_rnn import (
    Conv2dGruConv2d, 
    LAST_CN_NUM, LAST_H, LAST_W, IMG_CHANNEL, 
)
from Continue_ColorfulBall_Rnn256_45dimColor.train_config import CONFIG
from shared import *
from evalLinearity_shared import *
import rc_params

rc_params.init()

Z_RADIUS = 2
N_LATENT_DIM = 5

EXTENT = [-Z_RADIUS, Z_RADIUS, -Z_RADIUS, Z_RADIUS]

RAINBOW_EXP_GROUPS: List[ExpGroup] = [
    ours, beta_vae, 
    ae_1, ae_2, 
]

def main():
    plotNDims(3)
    # plotColor()
    # plotColorDisentangle(loadModel(RAINBOW_EXP_GROUPS[0]))

def hideTicks(ax: Axes):
    ax.tick_params(
        axis='both', which='both', 
        left=False, 
        bottom=False, 
        labelleft=False, 
        labelbottom=False, 
    )

def plotNDims(n=3):
    FIGSIZE = (11, 2.2)
    NECK_LINE_X = .02
    WIDTH_RATIOS = [.33, .02, .3, .02, .3]
    LEN_Z_LADDER = 7
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
                img = synth(model, z)
                ax.imshow(img, extent=EXTENT)
                hideTicks(ax)
                if expGroup_i == 0 and col_i == 0:
                    ax.set_ylabel(
                        '$z_%d$' % (row_i + 1), 
                        rotation=0, 
                    )
                    ax.yaxis.set_label_coords(-.4, .3)
                if row_i == n - 1:
                    if col_i % 3 == 0:
                        ax.set_xlabel(
                            '$%.1f$' % z_val, 
                        )
        subfig.suptitle(
            '(' + 'abc'[expGroup_i] + ') ' 
            + expGroup.display_name, 
        )
        # neckLine = Line2D(
        #     [NECK_LINE_X], 
        #     [0, 1], 
        #     color='k', linewidth=1.5, 
        # )
        # subfig.add_artist(neckLine)
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

def plotColor():
    RESOLUTION = 128
    Z_LADDER = torch.linspace(-Z_RADIUS, Z_RADIUS, RESOLUTION)
    LOC_ZERO = torch.zeros((3, ))

    fig, axes = plt.subplots(
        ncols=len(RAINBOW_EXP_GROUPS), 
        sharex=True, sharey=True, 
    )
    if len(RAINBOW_EXP_GROUPS) == 1:
        axes = [axes]   # for debug
    for (expGroup, ax) in zip(
        RAINBOW_EXP_GROUPS, axes, 
    ):
        model = loadModel(expGroup)
        canvas = torch.zeros((RESOLUTION, RESOLUTION, 3))
        for x, z_4 in tqdm([*enumerate(Z_LADDER)], expGroup.display_name):
            for y, z_5 in enumerate(Z_LADDER):
                img = synth(model, LOC_ZERO, torch.Tensor([z_4, z_5]))
                color = detectBallColor(img)
                canvas[x, y, :] = color
        ax.imshow(canvas, extent=EXTENT)
        ax.set_title(expGroup)
        ax.set_ylabel('$z_4$')
        ax.set_xlabel('$z_5$')
        hideTicks(ax)

    fig.suptitle('Detected color of the synthesized ball')
    plt.show()

def plotColorDisentangle(model):
    # RESOLUTION = 20
    # RESOLUTION = 64
    RESOLUTION = 256
    Z_LADDER = torch.linspace(-Z_RADIUS, Z_RADIUS, RESOLUTION)

    fig, axeses = plt.subplots(
        3, 3, 
        # sharex=True, sharey=True, 
    )
    SUBPLOT_PAD = .65
    fig.subplots_adjust(wspace=SUBPLOT_PAD, hspace=SUBPLOT_PAD)
    midAx = axeses[1][1]
    for row_i, axes in enumerate(axeses):
        for col_i, ax in enumerate(axes):
            default_z = torch.zeros((5, ))
            if ax is midAx:
                i, j = 3, 4
            else:
                i, j = 0, 2
                default_z[3] = row_i - 1
                default_z[4] = col_i - 1
            canvas = torch.zeros((RESOLUTION, RESOLUTION, 3))
            for x, z_i in tqdm(
                [*enumerate(Z_LADDER)], 
                f'({row_i}, {col_i}) / (3, 3)', 
            ):
                for y, z_j in enumerate(Z_LADDER):
                    z = default_z[:]
                    z[i] = z_i
                    z[j] = z_j
                    img = synth(model, z)
                    color = detectBallColor(img)
                    canvas[x, y, :] = color
            ax.imshow(canvas, extent=EXTENT)
            if ax is not midAx:
                drawCross(ax, 0, 0)
                drawCross(midAx, default_z[3], default_z[4])
            # ax.set_title('')
            ax.set_ylabel('$z_%d$' % (i + 1), rotation=0)
            ax.set_xlabel('$z_%d$' % (j + 1))
            # hideTicks(ax)

    fig.suptitle('Detected color of the synthesized ball')
    plt.show()

def drawCross(ax: Axes, x, y, c='k', radius=.15):
    ax.plot(
        [x - radius, x + radius], 
        [y - radius, y + radius], 
        c=c, linewidth=1, 
    )
    ax.plot(
        [x - radius, x + radius], 
        [y + radius, y - radius], 
        c=c, linewidth=1, 
    )

def detectBallColor(img: torch.Tensor):
    img_max = img.max(dim=2).values
    img_min = img.min(dim=2).values
    luminosity = (img_max + img_min) * .5
    saturation = (img_max - img_min) / (
        1 - (2 * luminosity - 1).abs()
    )
    metric = saturation + luminosity * .3
    # metric = luminosity
    argmax = metric.max() == metric
    return img[argmax, :][0, :]

main()
