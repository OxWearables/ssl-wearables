import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable

# CLASS_CODE = {'sleep': 0, 'awake': 1}
# CLASSES = ['sleep', 'awake']

CLASS_CODE = {"awake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
CLASSES = ["awake", "N1", "N2", "N3", "REM"]
# list of classes, ordered by code
NUM_CLASSES = len(CLASSES)


def encode_one_hot(y):
    """
    0 -> 1,0,0,0,0
    1 -> 0,1,0,0,0
    2 -> 0,0,1,0,0
    3 -> 0,0,0,1,0
    4 -> 0,0,0,0,1
    """
    return (y.reshape(-1, 1) == np.arange(NUM_CLASSES)).astype(int)


def get_hmm_para(model, data, cuda, batch_size=10000, num_workers=6):
    """Forward pass model on a dataset. Do this by batches so that we do
    not blow up the memory."""
    model.eval()

    mydata = DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    y_oob = []
    y_real = []
    with torch.no_grad():
        for my_x, my_y in mydata:
            my_x, my_y = Variable(my_x), Variable(my_y)
            my_x = my_x.float()
            my_y = my_y.float()
            if cuda:
                my_x, my_y = my_x.cuda(device=0), my_y.cuda(device=0)
            logits = model.classify(my_x)
            _, pred_idx = torch.max(logits, 1)
            y_oob.extend(logits.cpu().detach().numpy())
            y_real.extend(torch.max(my_y, 1)[1].data.cpu().detach().numpy())

    model.train()
    y_oob = np.stack(y_oob)
    y_real = np.stack(y_real)
    prior, emission, transition = train_hmm(y_oob, y_real)
    return prior, emission, transition


def train_hmm(Y_pred, y_true):
    """https://en.wikipedia.org/wiki/Hidden_Markov_model"""
    # Input Y_pred: N x 5
    #       y_true: N x 1
    # if Y_pred.ndim == 1 or Y_pred.shape[1] == 1:
    # Y_pred = encode_one_hot(Y_pred)

    prior = np.mean(y_true.reshape(-1, 1) == np.arange(NUM_CLASSES), axis=0)
    emission = np.vstack(
        [np.mean(Y_pred[y_true == i], axis=0) for i in range(NUM_CLASSES)]
    )
    transition = np.vstack(
        [
            np.mean(
                y_true[1:][(y_true == i)[:-1]].reshape(-1, 1)
                == np.arange(NUM_CLASSES),
                axis=0,
            )
            for i in range(NUM_CLASSES)
        ]
    )
    return prior, emission, transition


def viterbi(y_pred, prior, emission, transition):
    """https://en.wikipedia.org/wiki/Viterbi_algorithm"""
    small_number = 1e-16

    def log(x):
        return np.log(x + small_number)

    num_obs = len(y_pred)
    probs = np.zeros((num_obs, NUM_CLASSES))
    probs[0, :] = log(prior) + log(emission[:, y_pred[0]])
    for j in range(1, num_obs):
        for i in range(NUM_CLASSES):
            probs[j, i] = np.max(
                log(emission[i, y_pred[j]])
                + log(transition[:, i])
                + probs[j - 1, :]
            )  # probs already in log scale
    viterbi_path = np.zeros_like(y_pred)
    viterbi_path[-1] = np.argmax(probs[-1, :])
    for j in reversed(range(num_obs - 1)):
        viterbi_path[j] = np.argmax(
            log(transition[:, viterbi_path[j + 1]]) + probs[j, :]
        )  # probs already in log scale

    return viterbi_path
