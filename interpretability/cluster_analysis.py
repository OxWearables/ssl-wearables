# %%
import os

from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn import preprocessing
from sklearn import decomposition

import umap

# %%


def reduceme(X, method='umap'):
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
    reducer = umap.UMAP() if method == 'umap' else decomposition.PCA(n_components=2)
    X_red = reducer.fit_transform(X_scaled)
    return X_red


def scatter_plot(X, Y, ax, colors={}, title=None):
    NPOINTS_PER_LABEL = 200

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


def scatter_plot_all(Xs_and_titles, Y, colors={}, legend_ncol=4, borderaxespad=0, savename="scatterplot"):

    LATEX_WIDTH = 5.5  # inches
    NCOLS = len(Xs_and_titles)
    WIDTH = 1.25 * LATEX_WIDTH
    HEIGHT = 1.65 * WIDTH / 3

    fig, axs = plt.subplots(ncols=NCOLS, figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=0, right=1, bottom=.40, top=.93, wspace=.1, hspace=.1)

    for (X, title), ax in zip(Xs_and_titles, axs):
        scatter_plot(X, Y, ax, colors, title)

    handles = [mpatches.Patch(color=c, label=y.lower())
               for y, c in colors.items()]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=legend_ncol,
        frameon=False,
        borderaxespad=borderaxespad,
        fontsize=12,
        mode='expand',
    )

    fig.savefig(savename + '.pdf', format='pdf', dpi=100)

    return fig


# %%
DATA = {
    # A nice tool for color gradients:
    # https://hauselin.github.io/colorpalettejs/

    'adl': {
        'path': '/data/UKBB/ssl_downstream/adl_30hz_w10',
        # 'path': '/data/UKBB/ssl_downstream/adl_30hz_clean',
        'relabel': {
            'brush_teeth': 'brush teeth',
            'climb_stairs': 'climb stairs',
            'comb_hair': 'comb hair',
            'descend_stairs': 'descend stairs',
            'drink_glass': 'drink glass',
            'eat_meat': 'eat meat',
            'eat_soup': 'eat soup',
            'getup_bed': 'getup bed',
            'liedown_bed': 'liedown bed',
            'pour_water': 'pour water',
            'sitdown_chair': 'sitdown chair',
            'standup_chair': 'standup chair',
            'use_telephone': 'use telephone',
        },
        'colors': {
            'liedown bed': '#0d0887',
            'getup bed': '#370499',
            'sitdown chair': '#5801a4',
            'standup chair': '#7701a8',
            'eat meat': '#920fa3',
            'eat soup': '#ac2694',
            'pour water': '#c23c81',
            'drink glass': '#d45270',
            'brush teeth': '#e4695e',
            'comb hair': '#f1814d',
            'use telephone': '#fa9b3d',
            'walk': '#feb82c',
            'climb stairs': '#fad824',
            'descend stairs': '#f0f921',
        },
        'legend_ncol': 3,
        'borderaxespad': 0,
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
        'legend_ncol': 3,
        'borderaxespad': 2,
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
        'borderaxespad': 3,
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
            'climbing up': '#db5c68',
            'climbing down': '#f48849',
            'running': '#febd2a',
            'jumping': '#f0f921',
        },
        'legend_ncol': 4,
        'borderaxespad': 3,
    },

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
        'legend_ncol': 3,
        'borderaxespad': 0,
    },

    'rowlands': {
        # 'path': '/data/UKBB/rowlands_80hz_w10_o0',
        'path': '/data/UKBB/rowlands_o0_hang',
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
        'legend_ncol': 3,
        'borderaxespad': 0,
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
        'legend_ncol': 3,
        'borderaxespad': 3,
    },
}


# %%
XY_cached = {
}

# %%

plt.close('all')
for dataname, datainfo in tqdm(DATA.items()):

    if dataname in XY_cached:

        X_red, X_ssl_red, X_nossl_red, Y = XY_cached[dataname]

    else:

        X = np.load(os.path.join(datainfo['path'], 'X.npy'), mmap_mode='r')
        X_ssl = np.load(os.path.join(datainfo['path'], 'SSL_feats.npy'), mmap_mode='r')
        X_nossl = np.load(os.path.join(datainfo['path'], 'NoSSL_feats.npy'), mmap_mode='r')

        X = X.reshape(len(X), -1)
        X_ssl = X_ssl.reshape(len(X_ssl), -1)
        X_nossl = X_nossl.reshape(len(X_nossl), -1)

        Y = np.load(os.path.join(datainfo['path'], 'Y.npy'))
        Y = np.asarray([  # decode
            y if y not in datainfo['relabel']
            else datainfo['relabel'][y]
            for y in Y
        ])

        # subsample Capture-24 dataset as it's too large
        if dataname == 'capture24':
            unq, cnt = np.unique(Y, return_counts=True)
            n = cnt.min()
            idxs = []
            for y in unq:
                _idxs = np.where(Y == y)[0]
                _idxs = np.random.choice(_idxs, size=n, replace=False)
                idxs.append(_idxs)
            idxs = np.concatenate(idxs)
            X, X_ssl, X_nossl, Y = X[idxs], X_ssl[idxs], X_nossl[idxs], Y[idxs]

        X_red = reduceme(X)
        X_ssl_red = reduceme(X_ssl)
        X_nossl_red = reduceme(X_nossl)

        XY_cached[dataname] = (X_red, X_ssl_red, X_nossl_red, Y)

    os.makedirs(os.path.join("figs/umap", dataname), exist_ok=True)

    scatter_plot_all(
        [
            (X_red, 'Raw input'),
            (X_nossl_red, 'Features, no pretraining'),
            (X_ssl_red, 'Features, SSL pretraining'),
        ],
        Y,
        datainfo['colors'],
        datainfo['legend_ncol'],
        datainfo['borderaxespad'],
        os.path.join("figs/umap", dataname, f"umap_{dataname}")
    )
