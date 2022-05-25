#%%
import os
import sys
import pathlib
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import train_test_split
import umap

#%%

def reduceme(X, method='umap'):
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
    reducer = umap.UMAP() if method =='umap' else decomposition.PCA(n_components=2)
    X_red = reducer.fit_transform(X_scaled)
    return X_red


def scatter_plot(X, Y, ax, colors={}, title=None):
    NPOINTS_PER_LABEL = 1000

    for y in np.unique(Y):
        _X = X[Y == y]

        if len(_X) > NPOINTS_PER_LABEL:
            idxs = np.random.choice(len(_X), replace=False, size=NPOINTS_PER_LABEL)
            _X = _X[idxs]

        ax.scatter(_X[:, 0], _X[:, 1], c=colors.get(y, None), s=10, label=y, marker='.')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)


def scatter_plot_all(Xs_and_titles, Y, colors={}, legend_ncol=4, savename="scatterplot"):

    LATEX_WIDTH = 5.5  # inches
    NCOLS = len(Xs_and_titles)
    WIDTH = 1.55 * LATEX_WIDTH
    HEIGHT = 1.3 * WIDTH / 3

    fig, axs = plt.subplots(ncols=NCOLS, figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=0, right=1, bottom=.25, top=.93, wspace=.1, hspace=.1)

    for (X, title), ax in zip(Xs_and_titles, axs):
        scatter_plot(X, Y, ax, colors, title)

    handles = [mpatches.Patch(color=c, label=y.lower()) for y, c in colors.items()]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=legend_ncol, 
        fancybox=True, 
        borderaxespad=0
    )
    fig.savefig(savename + '.pdf', format='pdf')

    return fig


#%%
DATA = {
    'wisdm': {
        'path': '/data/UKBB/ssl_downstream/wisdm_30hz_clean',
        'relabel': {
            'dribbling': 'dribbling basketball',
            'kicking': 'kicking soccer ball',
            'catch': 'catch tennis ball',
            'pasta': 'eating pasta',
            'sandwich': 'eating sandwich',
            'chips': 'eating chips',
            'soup': 'eating soup',
            'drinking': 'drinking from cup',
            'teeth': 'brushing teeth',
            'folding': 'folding clothes',
        },
        'colors': {
            'sitting': '#0d0887',
            'typing': '#2f0596',
            'writing': '#4903a0',
            'eating pasta': '#6100a7',
            'eating sandwich': '#7801a8',
            'eating chips': '#8e0ca4',
            'eating soup': '#a21d9a',
            'drinking from cup': '#b42e8d',
            'standing': '#c43e7f',
            'brushing teeth': '#d24f71',
            'folding clothes': '#de6164',
            'clapping': '#e97257',
            'walking': '#f3854b',
            'stairs': '#f99a3e',
            'jogging': '#fdaf31',
            'dribbling basketball': '#fdc627',
            'kicking soccer ball': '#f8df25',
            'catch tennis ball': '#f0f921',
        },
        'legend_ncol': 5,
    },

    'adl': {
        'path': '/data/UKBB/ssl_downstream/adl_30hz_clean',
        'relabel': {
            'getup_bed': 'getup bed',
            'climb_stairs': 'climb stairs',
            'pour_water': 'pour water',
            'drink_glass': 'drink glass',
        },
        'colors': {
            'getup bed': '#0d0887',
            'walk': '#7e03a8',
            'climb stairs': '#cc4778',
            'pour water': '#f89540',
            'drink glass': '#f0f921',
        },
        'legend_ncol': 5,
    },

    'oppo': {
        'path': '/data/UKBB/oppo_33hz_w10_o5',
        'relabel': {
            3: "lying down",
            1: "standing",
            4: "sitting",
            2: "walking",
        },
        'colors': {
            'lying down': '#0d0887',
            'standing': '#9c179e',
            'sitting': '#ed7953',
            'walking': '#f0f921',
        },
        'legend_ncol': 4,
    },

    'pamap': {
        'path': '/data/UKBB/pamap_100hz_w10_o5',
        'relabel': {
            1: "lying down",
            2: "sitting",
            3: "standing",
            4: "walking",
            12: "ascending stairs",
            13: "descending stairs",
            16: "vacumm cleaning",
            17: "ironing",
        },
        'colors': {
            "lying down": '#0d0887',
            "sitting": '#5302a3',
            "standing": '#8b0aa5',
            "ironing": '#b83289',
            "vacumm cleaning": '#db5c68',
            "walking": '#f48849',
            "ascending stairs": '#febd2a',
            "descending stairs": '#f0f921',
        },
        'legend_ncol': 4,
    },

    'realworld': {
        'path': '/data/UKBB/ssl_downstream/realworld_30hz_clean',
        'relabel': {
            'lying': 'lying down',
            'climbingup': 'climbing up',
            'climbingdown': 'climbing down',
        },
        'colors': {
            'lying down': '#0d0887',
            'sitting': '#5302a3',
            'standing': '#8b0aa5',
            'walking': '#b83289',
            'climbingup': '#db5c68',
            'climbingdown': '#f48849',
            'running': '#febd2a',
            'jumping': '#f0f921',
        },
        'legend_ncol': 4,
    },

    'rowlands': {
        'path': '/data/UKBB/rowlands_80hz_w10_o5',
        'relabel': {
            'Lying': 'Lying down',
        },
        'colors': {
            'Lying down': '#0d0887',
            'Standing': '#3a049a',
            'Seated Computer Work': '#5c01a6',
            'Shelf Stacking': '#7e03a8',
            'Window Washing': '#9c179e',
            'Washing Up': '#b52f8c',
            'Sweeping': '#cc4778',
            'Stairs': '#de5f65',
            '4km/hr Walk': '#ed7953',
            '5km/hr Walk': '#f89540',
            '6km/hr Walk': '#fdb42f',
            '8km/hr Run': '#fbd524',
            '10+km/hr Run': '#f0f921',
        },
        'legend_ncol': 5,
    },

    'capture24': {
        'path': '/data/UKBB/capture24_30hz_w10_o0',
        'relabel': {},
        'colors': {
            'sit-stand': '#0d0887',
            'vehicle': '#b12a90',
            'mixed': '#e16462',
            'walking': '#fca636',
            'bicycling': '#f0f921',
        },
        'legend_ncol': 5,
    },
}

#%%

plt.close('all')
for dataname, datainfo in DATA.items():
    X = np.load(os.path.join(datainfo['path'], 'X.npy'))
    X_ssl = np.load(os.path.join(datainfo['path'], 'SSL_feats.npy'))
    X_nossl = np.load(os.path.join(datainfo['path'], 'NoSSL_feats.npy'))

    X = X.reshape(len(X), -1)
    X_ssl = X_ssl.reshape(len(X_ssl), -1)
    X_nossl = X_nossl.reshape(len(X_nossl), -1)

    Y = np.load(os.path.join(datainfo['path'], 'Y.npy'))
    Y = np.asarray([  # decode
        y if y not in datainfo['relabel']
        else datainfo['relabel'][y]
        for y in Y
    ])

    NMAX = 20000
    SEED = 42
    if len(X) > NMAX:  # subsample if too large
        # Note: Use same seed for all!
        _, X = train_test_split(X, test_size=NMAX, random_state=SEED, stratify=Y)
        _, X_ssl = train_test_split(X_ssl, test_size=NMAX, random_state=SEED, stratify=Y)
        _, X_nossl = train_test_split(X_nossl, test_size=NMAX, random_state=SEED, stratify=Y)
        _, Y = train_test_split(Y, test_size=NMAX, random_state=SEED, stratify=Y)

    X_red = reduceme(X)
    X_ssl_red = reduceme(X_ssl)
    X_nossl_red = reduceme(X_nossl)

    os.makedirs(os.path.join("figs/umap", dataname), exist_ok=True)

    scatter_plot_all(
        [
            (X_red, 'Raw input'),
            (X_nossl_red, 'Features without pretraining'),
            (X_ssl_red, 'Features with SSL pretraining'),
        ],
        Y,
        datainfo['colors'],
        datainfo['legend_ncol'],
        os.path.join("figs/umap", dataname, f"umap_{dataname}")
    )
# %%
