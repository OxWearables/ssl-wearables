import torch
from torch.autograd import Variable
import json
import numpy as np
import sklearn.metrics as metrics


def trans30two1(data):
    # transform a period of 30 seconds to one sec per row
    x = data[0, :]
    y = data[1, :]
    z = data[2, :]

    x = x.reshape(30, 1, 30)
    y = y.reshape(30, 1, 30)
    z = z.reshape(30, 1, 30)

    data = np.concatenate((x, y, z), axis=1)
    return data


def load_weights_dist2norm(model, model_path):
    # original saved file with DataParallel
    state_dict = torch.load(model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def compute_scores(y_true, y_pred):
    """Compute a bunch of scoring functions"""
    confusion = metrics.confusion_matrix(y_true, y_pred)
    per_class_recall = metrics.recall_score(y_true, y_pred, average=None)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    balanced_acuracy = metrics.balanced_accuracy_score(y_true, y_pred)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    return {
        "confusion": confusion,
        "per_class_recall": per_class_recall,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acuracy,
        "kappa": kappa,
    }


def print_scores(scores):
    print("Accuracy score:", scores["accuracy"])
    print("Balanced accuracy score:", scores["balanced_accuracy"])
    print("Cohen kappa score:", scores["kappa"])
    print("\nPer-class recall scores:")
    print(
        "sleep      : {}\n" "awake   : {}".format(*scores["per_class_recall"])
    )
    print("\nConfusion matrix:\n", scores["confusion"])


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_scores(dest_file, scores):
    dumped = json.dumps(scores, cls=NumpyEncoder)

    with open(dest_file, "w") as f:
        json.dump(dumped, f)


def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """

    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())


def gradient_clip_lstm(model, clip_value=1):
    for name, param in model.named_parameters():
        if name[:4] == "lstm":
            param.register_hook(
                lambda grad: torch.clamp(grad, -clip_value, clip_value)
            )


def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """

    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y

    return encode


def get_one_hot(targets, nb_classes):
    """numpy version"""
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied,
        e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return (
        torch.log(
            sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8
        )
        + max
    )
