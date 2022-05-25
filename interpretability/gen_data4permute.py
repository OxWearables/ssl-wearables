import os
from os.path import expanduser

home = expanduser("~")

import numpy as np
import pickle

from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
from interpretability.attribution import forward_by_batches


import hydra

# PyTorch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import pathlib
from torchsummary import summary
import sslearning

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from sslearning.models.accNet import SSLNET, Resnet
from sslearning.utils import load_weights_dist2norm, trans30two1
from downstream_task_evaluation import load_weights
from sslearning.data.data_loader import subject_dataset

import gc

cuda = torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()
cudnn.benchmark = True


def set_seed():
    # For reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cudnn.benchmark = True
    if cuda:
        torch.cuda.manual_seed_all(random_seed)


set_seed()
device = torch.device("cpu")

#Python ≥ 3.5, see: https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory

config_dir = pathlib.Path('../conf/')
hydra.initialize(config_path=config_dir)
cfg=compose(config_name="config.yaml")
cfg=compose(overrides= ["+model=resnet", "+dataloader=default", "+task=time_reversal", "++task.positive_ratio=0.5"])
cfg.dataloader.epoch_len=10
print(cfg)


# Uncomment this for Andrew
model_name='final_model_10.mdl'
#model_path = home + '/data/ssl/' + model_name
model_path = '/data/UKBB/final_models/100k_epoch_30.mdl'

model = Resnet(output_size=2, cfg=cfg)
load_weights_dist2norm(model, model_path)
model.eval()


from sslearning.data.datautils import Transform
from sslearning.data.data_transformation import flip
transform=Transform(transformations=['rescale'], channel_wise=False, limits=[-1, 1])

import glob
#  1. Load dataset
data_path = '/data/UKBB/SSL/100k/test/processed_group8/data/*.npy'
file_list = glob.glob(data_path)
num_sub = 1000

'''determine exemplary samples'''
#1. predicted correctly for forward and reversed samples
#2. high posterior probabilty for each prediction (>threshold)

def get_data4one(sample_path):
    sample_path = file_list[0]
    dataset = subject_dataset(sample_path, has_std=True)
    data_generator=DataLoader(dataset, batch_size=64, shuffle=False)
    Y, Yfit, _, prob=forward_by_batches(model, data_generator, device=device)


    X_data=dataset.X.cpu().detach().numpy()

    threshold=0.85

    all_mask=(Y==Yfit) & (np.max(prob, axis=1)>threshold)
    print('Total # exemplary examples: {:} ({:.2f}%)'.format(all_mask.sum(), 100*all_mask.sum()/len(all_mask)))

    fwd_idx=np.arange(0, len(X_data), 2)
    rev_idx=np.arange(1, len(X_data), 2)

    X_fwd=X_data[fwd_idx]
    X_rev=X_data[rev_idx]

    fwd_mask=(Y[fwd_idx]==Yfit[fwd_idx]) & (np.max(prob[fwd_idx], axis=1)>threshold)
    rev_mask=(Y[rev_idx]==Yfit[rev_idx]) & (np.max(prob[rev_idx], axis=1)>threshold)

    print('Total # exem plary (forward) examples:', fwd_mask.sum())
    print('Total # exemplary (reverse) examples:', rev_mask.sum())

    mask=(fwd_mask==True) & (rev_mask==True)

    print('Total # exemplary (matched) examples: {:} ({:.2f}%)'.format(mask.sum(), 100*mask.sum()/len(mask)))

    indexs=np.where(mask)[0]
    X_fwd=X_fwd[indexs]
    X_rev=X_rev[indexs]
    return X_fwd, X_rev

X_ford = []
X_back = []
i = 0
for my_file in file_list:
    if i == 100:
        break

    x_forward, x_backward = get_data4one(my_file)
    if i == 0:
        X_ford = x_forward
        X_back = x_backward
    else:
        #print(X_ford.shape)
        #print(x_forward.shape)
        X_ford = np.concatenate((X_ford, x_forward), axis=0)
        X_back = np.concatenate((X_back, x_backward), axis=0)

    if i % 10 == 0:
        print("Processing file %d" % i)
    i += 1

print(X_ford.shape)
print(X_back.shape)

np.save('/data/UKBB/SSL/lrp_data/X_f.npy', X_ford)
np.save('/data/UKBB/SSL/lrp_data/X_b.npy', X_back)


