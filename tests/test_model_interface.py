import torch
import numpy as np


def test_model_basic():
    repo = "OxWearables/ssl-wearables"
    harnet10 = torch.hub.load(repo, "harnet10", class_num=5, pretrained=True)
    x = np.random.rand(1, 3, 300)
    x = torch.FloatTensor(x)
    y = harnet10(x)

    assert y.shape[1] == 5


def test_model_90():
    repo = "OxWearables/ssl-wearables"
    harnet10 = torch.hub.load(repo, "harnet30", class_num=5, pretrained=True)
    x = np.random.rand(1, 3, 900)
    x = torch.FloatTensor(x)
    y = harnet10(x)

    assert y.shape[1] == 5


def test_model_default():
    repo = "OxWearables/ssl-wearables"
    harnet10 = torch.hub.load(repo, "harnet10")
    x = np.random.rand(1, 3, 300)
    x = torch.FloatTensor(x)
    y = harnet10(x)
    assert y.shape[1] == 2
