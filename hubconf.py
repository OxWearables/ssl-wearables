import os
from sslearning.models.accNet import Resnet
import torch
import copy

dependencies = ["torch"]


def load_weights(
    weight_path, model, my_device="cpu", name_start_idx=2, is_dist=False
):
    # only need to change weights name when the
    # model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names

    if is_dist:
        for key in pretrained_dict:
            para_names = key.split(".")
            new_key = ".".join(para_names[name_start_idx:])
            pretrained_dict_v2[new_key] = pretrained_dict_v2.pop(key)

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))


def harnet10(pretrained=False, my_device="cpu", class_num=2, **kwargs):
    """
    harnet10 model
    pretrained (bool): kwargs, load pretrained weights into the model

    Input:

    X is of size: N x 3 x 300. N is the number of examples.
    3 is the xyz channel. 300 consists of
    a 10 second recording with 30hz.

    Output:
    my_device (str)
    class_num (int): the number of classes to predict

    Example:
    repo = 'OxWearables/ssl-wearables'
    model = torch.hub.load(repo, 'harnet10',
                           pretrained=True)
    x = np.random.rand(1, 3, 300)
    x = torch.FloatTensor(x)
    model(x)

    """
    # Call the model, load pretrained weights
    model = Resnet(
        output_size=class_num,
        is_eva=True,
        resnet_version=1,
    )

    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(
            dirname, "", "model_check_point", "mtl_best.mdl"
        )
        load_weights(
            checkpoint, model, my_device, is_dist=True, name_start_idx=1
        )

    return model


def harnet30(pretrained=False, my_device="cpu", class_num=2, **kwargs):
    """
    harnet10 model
    pretrained (bool): kwargs, load pretrained weights into the model

    Input:

    X is of size: N x 3 x 300. N is the number of examples.
    3 is the xyz channel. 300 consists of
    a 10 second recording with 30hz.

    Output:
    my_device (str)
    class_num (int): the number of classes to predict

    Example:
    repo = 'OxWearables/ssl-wearables'
    model = torch.hub.load(repo, 'harnet10',
                           pretrained=True)
    x = np.random.rand(1, 3, 300)
    x = torch.FloatTensor(x)
    model(x)

    """
    # Call the model, load pretrained weights
    model = Resnet(
        output_size=class_num,
        is_eva=True,
        epoch_len=30,
        resnet_version=1,
    )

    if pretrained:
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(
            dirname, "", "model_check_point", "mtl_30_best.mdl"
        )
        load_weights(
            checkpoint, model, my_device, is_dist=True, name_start_idx=1
        )

    return model
