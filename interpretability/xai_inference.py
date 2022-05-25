"""
This scripts loads a file and makes predictions in a multi-task fashion. Used only for XAI experiments to understand
the embedding space. For fine-tuning on new datasets, please refer to `downstream_task_evaluation.py` instead.

* Input: path to an numpy array of size N x 3 x 300
* Output: AoT and TimeW labels of size N x 2 named `ssl_pred.numpy`

Usage: python xai_inference.py

"""

import copy
import numpy as np

# Model utils
from sslearning.models.accNet import  Resnet

# Data utils

# Torch
import torch
from torch.autograd import Variable

# Plotting
from datetime import datetime

cuda = torch.cuda.is_available()
now = datetime.now()


################################
#
#
#       helper functions
#
#
################################
def set_seed(my_seed=0):
    random_seed = my_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed_all(random_seed)


def set_up_data4train(my_X, aot_y, scale_y, permute_y, time_w_y, cfg, my_device):
    aot_y, scale_y, permute_y, time_w_y = Variable(aot_y), Variable(scale_y), \
                                          Variable(permute_y), Variable(time_w_y)
    my_X = Variable(my_X)
    my_X = my_X.to(my_device, dtype=torch.float)
    aot_y = aot_y.to(my_device, dtype=torch.long)
    scale_y = scale_y.to(my_device, dtype=torch.long)
    permute_y = permute_y.to(my_device, dtype=torch.long)
    time_w_y = time_w_y.to(my_device, dtype=torch.long)
    return my_X, aot_y, scale_y, permute_y, time_w_y


def evaluate_model(model, data_loader, my_device, cfg):
    model.eval()
    aot_real, aot_pred = []
    time_w_real, time_w_pred = []
    for i, (my_X, aot_y, scale_y, permute_y, time_w_y) in enumerate(data_loader):
        with torch.no_grad():
            my_X, aot_y, _, _, time_w_y = set_up_data4train(my_X, aot_y, scale_y, permute_y, time_w_y,
                                                                          cfg, my_device)
            # print the expected input lengths
            aot_y_pred_logits, _, _, time_w_h_pred = model(my_X)
            pred_y_aot = torch.argmax(aot_y_pred_logits, dim=1)
            pred_y_time_w = torch.argmax(time_w_h_pred, dim=1)

            aot_real.extend(aot_y)
            aot_pred.extend(pred_y_aot)
            time_w_real.extend(time_w_y)
            aot_pred.extend(pred_y_time_w)
    return aot_real, aot_pred, time_w_real, aot_pred


def load_weights(weight_path, model, my_device, name_start_idx=2, is_dist=False):
    # only need to change weights name when the model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(pretrained_dict)  # v2 has the right para names

    if is_dist:
        for key in pretrained_dict:
            para_names = key.split('.')
            new_key = '.'.join(para_names[name_start_idx:])
            pretrained_dict_v2[new_key] = pretrained_dict_v2.pop(key)

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {k: v for k, v in pretrained_dict_v2.items() if k in model_dict
                       and k.split('.')[0] != 'classifier'}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))


def main():
    set_seed()
    GPU = 1 # change this for cpu
    network_weight_path = "/data/UKBB/SSL/final_models/mtl_10_final.mdl" # change this for network

    if GPU >= -1:
        my_device = 'cuda:' + str(GPU)
    else:
        my_device = 'cpu'

    model = Resnet(output_size=2, is_mtl=True)
    model = model.float()
    print(model)

    print("Training using device %s" % my_device)
    model.to(my_device, dtype=torch.float)

    load_weights(network_weight_path, model, my_device, is_dist=True, name_start_idx=1)

    model.eval()
    N = 100
    my_X = torch.ones([N, 3, 300], dtype=torch.float64)
    my_X = Variable(my_X)
    my_X = my_X.to(my_device, dtype=torch.float)

    # create a torch tensor of size 3k x 3 x 300
    aot_y_pred_logits, _, _, time_w_h_pred = model(my_X)
    pred_y_aot = torch.argmax(aot_y_pred_logits, dim=1)
    pred_y_time_w = torch.argmax(time_w_h_pred, dim=1)

    # 0 original, 1 transformed
    print('Aot prediction')
    print(pred_y_aot)
    print('Time_W prediction')
    print(pred_y_time_w)


if __name__ == "__main__":
    main()
