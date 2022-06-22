import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.signal as signal
from scipy.interpolate import interp1d
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import hydra
from omegaconf import OmegaConf
from torchvision import transforms
import pathlib

# SSL net
from sslearning.models.accNet import cnn1, SSLNET, Resnet
from sslearning.scores import classification_scores, classification_report
import copy
from sklearn import preprocessing
from sslearning.data.data_loader import NormalDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from sslearning.pytorchtools import EarlyStopping
from sslearning.data.datautils import RandomSwitchAxis, RotationAxis
import torch
import torch.nn as nn
import logging
from datetime import datetime
import collections
from hydra.utils import get_original_cwd
import shutil
import joblib

"""
python downstream_task_evaluation.py -m data=rowlands_10s,oppo_10s
report_root='/home/cxx579/ssw/reports/mtl/aot'
is_dist=false gpu=0 model=resnet evaluation=mtl_1k_ft evaluation.task_name=aot
"""


def train_val_split(X, Y, group, val_size=0.125, fold_id=0):
    num_split = 1
    folds = GroupShuffleSplit(
        num_split, test_size=val_size, random_state=41
    ).split(X, Y, groups=group)
    train_idx, val_idx = next(folds)

    train_id_path = os.path.join(
        "/data/UKBB/SSL/ssl_cv_models", str(fold_id), "train_pid.npy"
    )
    val_id_path = os.path.join(
        "/data/UKBB/SSL/ssl_cv_models", str(fold_id), "val_pid.npy"
    )

    np.save(train_id_path, train_idx)
    np.save(val_id_path, val_idx)

    return X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm1d") != -1:
        m.eval()


def freeze_weights(model):
    i = 0
    # Set Batch_norm running stats to be frozen
    # Only freezing ConV layers for now
    # or it will lead to bad results
    # http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    for name, param in model.named_parameters():
        if name.split(".")[0] == "feature_extractor":
            param.requires_grad = False
            i += 1
    print("Weights being frozen: %d" % i)
    model.apply(set_bn_eval)


def evaluate_model(model, data_loader, my_device, loss_fn, cfg):
    model.eval()
    losses = []
    acces = []
    for i, (my_X, my_Y) in enumerate(data_loader):
        with torch.no_grad():
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            if cfg.data.task_type == "regress":
                true_y = my_Y.to(my_device, dtype=torch.float)
            else:
                true_y = my_Y.to(my_device, dtype=torch.long)

            logits = model(my_X)
            loss = loss_fn(logits, true_y)

            pred_y = torch.argmax(logits, dim=1)

            test_acc = torch.sum(pred_y == true_y)
            test_acc = test_acc / (list(pred_y.size())[0])

            losses.append(loss.cpu().detach().numpy())
            acces.append(test_acc.cpu().detach().numpy())
    losses = np.array(losses)
    acces = np.array(acces)
    return np.mean(losses), np.mean(acces)


def get_class_weights(y):
    # obtain inverse of frequency as weights for the loss function
    counter = collections.Counter(y)
    for i in range(len(counter)):
        if i not in counter.keys():
            counter[i] = 1

    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    print("Weight tensor: ")
    print(weights)
    return weights


def setup_data(train_idxs, test_idxs, X_feats, Y, groups, cfg, fold_id):
    tmp_X_train, X_test = X_feats[train_idxs], X_feats[test_idxs]
    tmp_Y_train, Y_test = Y[train_idxs], Y[test_idxs]
    group_train, group_test = groups[train_idxs], groups[test_idxs]

    # when we are not using all the subjects
    if cfg.data.subject_count != -1:
        tmp_X_train, tmp_Y_train, group_train = get_data_with_subject_count(
            cfg.data.subject_count, tmp_X_train, tmp_Y_train, group_train
        )

    test_id_path = os.path.join(
        "/data/UKBB/SSL/ssl_cv_models", str(fold_id), "train_pid.npy"
    )
    np.save(test_id_path, test_idxs)

    # When changing the number of training data, we
    # will keep the test data fixed
    if cfg.data.held_one_subject_out:
        folds = LeaveOneGroupOut().split(
            tmp_X_train, tmp_Y_train, groups=group_train
        )
        folds = list(folds)
        final_train_idxs, final_val_idxs = folds[0]
        X_train, X_val = (
            tmp_X_train[final_train_idxs],
            tmp_X_train[final_val_idxs],
        )
        Y_train, Y_val = (
            tmp_Y_train[final_train_idxs],
            tmp_Y_train[final_val_idxs],
        )
    else:
        # We further divide up train into 70/10 train/val split
        X_train, X_val, Y_train, Y_val = train_val_split(
            tmp_X_train, tmp_Y_train, group_train, fold_id
        )

    my_transform = None
    if cfg.augmentation:
        my_transform = transforms.Compose([RandomSwitchAxis(), RotationAxis()])
    train_dataset = NormalDataset(
        X_train, Y_train, name="train", isLabel=True, transform=my_transform
    )
    val_dataset = NormalDataset(X_val, Y_val, name="val", isLabel=True)
    test_dataset = NormalDataset(
        X_test, Y_test, pid=group_test, name="test", isLabel=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.evaluation.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.evaluation.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.evaluation.num_workers,
    )

    weights = []
    if cfg.data.task_type == "classify":
        weights = get_class_weights(Y_train)
    return train_loader, val_loader, test_loader, weights


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        if len(yhat.size()) == 2:
            yhat = yhat.flatten()
        # return torch.sqrt(self.mse(yhat, y))
        return self.mse(yhat, y)


def train_mlp(model, train_loader, val_loader, cfg, my_device, weights):
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.evaluation.learning_rate, amsgrad=True
    )

    if cfg.data.task_type == "classify":
        if cfg.data.weighted_loss_fn:
            weights = torch.FloatTensor(weights).to(my_device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = RMSELoss()

    early_stopping = EarlyStopping(
        patience=cfg.evaluation.patience, path=cfg.model_path, verbose=True
    )
    for epoch in range(cfg.evaluation.num_epoch):
        model.train()
        train_losses = []
        train_acces = []
        for i, (my_X, my_Y) in enumerate(train_loader):
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            if cfg.data.task_type == "regress":
                true_y = my_Y.to(my_device, dtype=torch.float)
            else:
                true_y = my_Y.to(my_device, dtype=torch.long)

            logits = model(my_X)
            loss = loss_fn(logits, true_y)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            pred_y = torch.argmax(logits, dim=1)
            train_acc = torch.sum(pred_y == true_y)
            train_acc = train_acc / (pred_y.size()[0])

            train_losses.append(loss.cpu().detach().numpy())
            train_acces.append(train_acc.cpu().detach().numpy())
        val_loss, val_acc = evaluate_model(
            model, val_loader, my_device, loss_fn, cfg
        )

        epoch_len = len(str(cfg.evaluation.num_epoch))
        print_msg = (
            f"[{epoch:>{epoch_len}}/{cfg.evaluation.num_epoch:>{epoch_len}}] "
            + f"train_loss: {np.mean(train_losses):.5f} "
            + f"valid_loss: {val_loss:.5f}"
        )
        early_stopping(val_loss, model)
        print(print_msg)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model


def mlp_predict(model, data_loader, my_device, cfg):
    predictions_list = []
    true_list = []
    pid_list = []
    model.eval()
    for i, (my_X, my_Y, my_PID) in enumerate(data_loader):
        with torch.no_grad():
            my_X, my_Y = Variable(my_X), Variable(my_Y)
            my_X = my_X.to(my_device, dtype=torch.float)
            if cfg.data.task_type == "regress":
                true_y = my_Y.to(my_device, dtype=torch.float)
                pred_y = model(my_X)
            else:
                true_y = my_Y.to(my_device, dtype=torch.long)
                logits = model(my_X)
                pred_y = torch.argmax(logits, dim=1)

            true_list.append(true_y.cpu())
            predictions_list.append(pred_y.cpu())
            pid_list.extend(my_PID)
    true_list = torch.cat(true_list)
    predictions_list = torch.cat(predictions_list)
    return (
        torch.flatten(true_list).numpy(),
        torch.flatten(predictions_list).numpy(),
        np.array(pid_list),
    )


def init_model(cfg, my_device):
    if cfg.model.resnet_version > 0:
        model = Resnet(
            output_size=cfg.data.output_size,
            is_eva=True,
            resnet_version=cfg.model.resnet_version,
            epoch_len=cfg.dataloader.epoch_len,
        )
    else:
        model = SSLNET(
            output_size=cfg.data.output_size, flatten_size=1024
        )  # VGG
    model.to(my_device, dtype=torch.float)
    return model


def setup_model(cfg, my_device):
    model = init_model(cfg, my_device)

    if cfg.evaluation.load_weights:
        load_weights(
            cfg.evaluation.flip_net_path,
            model,
            my_device,
            is_dist=cfg.is_dist,
            name_start_idx=1,
        )
    if cfg.evaluation.freeze_weight:
        freeze_weights(model)
    return model


def get_train_test_split(cfg, X_feats, y, groups):
    # support leave one subject out and split by proportion
    if cfg.data.held_one_subject_out:
        folds = LeaveOneGroupOut().split(X_feats, y, groups=groups)
    else:
        # Train-test multiple times with a 80/20 random split each
        folds = GroupShuffleSplit(
            cfg.num_split, test_size=0.2, random_state=42
        ).split(X_feats, y, groups=groups)
    return folds


def train_test_mlp(
    train_idxs, test_idxs, X_feats, y, groups, cfg, my_device, fold_id=0
):
    model = setup_model(cfg, my_device)
    if cfg.is_verbose:
        print(model)
    train_loader, val_loader, test_loader, weights = setup_data(
        train_idxs, test_idxs, X_feats, y, groups, cfg, fold_id
    )
    train_mlp(model, train_loader, val_loader, cfg, my_device, weights)

    model = init_model(cfg, my_device)

    model.load_state_dict(torch.load(cfg.model_path))

    cv_model_path = os.path.join(
        "/data/UKBB/SSL/ssl_cv_models",
        str(fold_id),
        "cnn_pretrained_" + str(cfg.evaluation.freeze_weight) + ".mdl",
    )
    shutil.copy2(cfg.model_path, cv_model_path)

    y_test, y_test_pred, pid_test = mlp_predict(
        model, test_loader, my_device, cfg
    )

    # save this for every single subject
    my_pids = np.unique(pid_test)
    results = []
    for current_pid in my_pids:
        subject_filter = current_pid == pid_test
        subject_true = y_test[subject_filter]
        subject_pred = y_test_pred[subject_filter]

        result = classification_scores(subject_true, subject_pred)
        results.append(result)
    return results


def evaluate_mlp(X_feats, y, cfg, my_device, logger, groups=None):
    """Train a random forest with X_feats and Y.
    Report a variety of performance metrics based on multiple runs."""

    if cfg.data.task_type == "classify":
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
    else:
        y = y * 1.0

    if isinstance(X_feats, pd.DataFrame):
        X_feats = X_feats.to_numpy()

    folds = get_train_test_split(cfg, X_feats, y, groups)

    results = []
    fold_id = 0
    for train_idxs, test_idxs in folds:
        result = train_test_mlp(
            train_idxs,
            test_idxs,
            X_feats,
            y,
            groups,
            cfg,
            my_device,
            fold_id=fold_id,
        )
        results.extend(result)

    pathlib.Path(cfg.report_root).mkdir(parents=True, exist_ok=True)
    classification_report(results, cfg.report_path)


def train_test_rf(
    train_idxs,
    test_idxs,
    X_feats,
    Y,
    cfg,
    groups,
    task_type="classify",
    fold_id=0,
):
    X_train, X_test = X_feats[train_idxs], X_feats[test_idxs]
    Y_train, Y_test = Y[train_idxs], Y[test_idxs]
    group_train, group_test = groups[train_idxs], groups[test_idxs]

    # when we are not using all the subjects
    if cfg.data.subject_count != -1:
        X_train, Y_train, group_train = get_data_with_subject_count(
            cfg.data.subject_count, X_train, Y_train, group_train
        )
    if task_type == "classify":
        model = BalancedRandomForestClassifier(
            n_estimators=3000,
            replacement=True,
            sampling_strategy="not minority",
            n_jobs=1,
            random_state=42,
        )
    elif task_type == "regress":
        model = RandomForestRegressor(
            n_estimators=200,  # more is too expensive
            n_jobs=1,
            random_state=42,
            max_features=0.333,
        )

    model.fit(X_train, Y_train)
    Y_test_pred = model.predict(X_test)

    rf_path = os.path.join(
        "/data/UKBB/SSL/ssl_cv_models", str(fold_id), "rf.joblib"
    )
    joblib.dump(model, rf_path)

    results = []
    for current_pid in np.unique(group_test):
        subject_filter = group_test == current_pid
        subject_true = Y_test[subject_filter]
        subject_pred = Y_test_pred[subject_filter]

        result = classification_scores(subject_true, subject_pred)
        results.append(result)

    return results


def evaluate_feats(X_feats, Y, cfg, logger, groups=None, task_type="classify"):
    """Train a random forest with X_feats and Y.
    Report a variety of performance metrics based on multiple runs."""

    if isinstance(X_feats, pd.DataFrame):
        X_feats = X_feats.to_numpy()

    # Train-test multiple times with a 80/20 random split each
    # Five-fold or Held one subject out
    folds = get_train_test_split(cfg, X_feats, Y, groups)
    print("loading done")
    results = Parallel(n_jobs=1)(
        delayed(train_test_rf)(
            train_idxs, test_idxs, X_feats, Y, cfg, groups, task_type, fold_idx
        )
        for fold_idx, (train_idxs, test_idxs) in enumerate(folds)
    )
    results = np.array(results)

    results = np.array(
        [
            fold_result
            for fold_results in results
            for fold_result in fold_results
        ]
    )

    print(results)
    pathlib.Path(cfg.report_root).mkdir(parents=True, exist_ok=True)
    classification_report(results, cfg.report_path)


def handcraft_features(xyz, sample_rate):
    """Our baseline handcrafted features. xyz is a window of shape (N,3)"""

    feats = {}
    feats["xMean"], feats["yMean"], feats["zMean"] = np.mean(xyz, axis=0)
    feats["xStd"], feats["yStd"], feats["zStd"] = np.std(xyz, axis=0)
    feats["xRange"], feats["yRange"], feats["zRange"] = np.ptp(xyz, axis=0)

    x, y, z = xyz.T

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # ignore div by 0 warnings
        feats["xyCorr"] = np.nan_to_num(np.corrcoef(x, y)[0, 1])
        feats["yzCorr"] = np.nan_to_num(np.corrcoef(y, z)[0, 1])
        feats["zxCorr"] = np.nan_to_num(np.corrcoef(z, x)[0, 1])

    m = np.linalg.norm(xyz, axis=1)

    feats["mean"] = np.mean(m)
    feats["std"] = np.std(m)
    feats["range"] = np.ptp(m)
    feats["mad"] = stats.median_abs_deviation(m)
    feats["kurt"] = stats.kurtosis(m)
    feats["skew"] = stats.skew(m)
    feats["enmomean"] = np.mean(np.abs(m - 1))

    # Spectrum using Welch's method with 3s segment length
    # First run without detrending to get the true spectrum
    freqs, powers = signal.welch(
        m,
        fs=sample_rate,
        nperseg=3 * sample_rate,
        noverlap=2 * sample_rate,
        detrend=False,
        average="median",
    )

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # ignore div by 0 warnings
        feats["pentropy"] = np.nan_to_num(stats.entropy(powers + 1e-16))

    # Spectrum using Welch's method with 3s segment length
    # Now do detrend to find dominant freqs
    freqs, powers = signal.welch(
        m,
        fs=sample_rate,
        nperseg=3 * sample_rate,
        noverlap=2 * sample_rate,
        detrend="constant",
        average="median",
    )

    peaks, _ = signal.find_peaks(powers)
    peak_powers = powers[peaks]
    peak_freqs = freqs[peaks]
    peak_ranks = np.argsort(peak_powers)[::-1]
    if len(peaks) >= 2:
        feats["f1"] = peak_freqs[peak_ranks[0]]
        feats["f2"] = peak_freqs[peak_ranks[1]]
    elif len(peaks) == 1:
        feats["f1"] = feats["f2"] = peak_freqs[peak_ranks[0]]
    else:
        feats["f1"] = feats["f2"] = 0

    return feats


def forward_by_batches(cnn, X, cnn_input_size, my_device="cpu"):
    """Forward pass model on a dataset. Includes resizing to model input size.
    Do this by batches so that we don't blow up the memory.
    """

    BATCH_SIZE = 1024

    X_feats = []
    cnn.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(X), BATCH_SIZE)):
            batch_end = i + BATCH_SIZE
            X_batch = X[i:batch_end]

            # Resize to expected input length
            X_batch = resize(X_batch, length=cnn_input_size)
            X_batch = X_batch.astype("f4")  # PyTorch defaults to float32
            X_batch = np.transpose(
                X_batch, (0, 2, 1)
            )  # channels first: (N,M,3) -> (N,3,M) channel first format
            X_batch = torch.from_numpy(X_batch)
            X_batch = X_batch.to(my_device, dtype=torch.float)

            if my_device == "cpu":
                X_feats.append(cnn(X_batch).numpy())
            else:
                X_feats.append(cnn(X_batch).cpu().numpy())

    X_feats = np.concatenate(X_feats)

    return X_feats


def resize(X, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """

    length_orig = X.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    X = interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )

    return X


def get_data_with_subject_count(subject_count, X, y, pid):
    subject_list = np.unique(pid)

    if subject_count == len(subject_list):
        valid_subjects = subject_list
    else:
        valid_subjects = subject_list[:subject_count]

    pid_filter = [my_subject in valid_subjects for my_subject in pid]

    filter_X = X[pid_filter]
    filter_y = y[pid_filter]
    filter_pid = pid[pid_filter]
    return filter_X, filter_y, filter_pid


def load_weights(
    weight_path, model, my_device, name_start_idx=2, is_dist=False
):
    # only need to change weights name when
    # the model is trained in a distributed manner

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


@hydra.main(config_path="conf", config_name="config_eva")
def main(cfg):
    """Evaluate hand-crafted vs deep-learned features"""

    logger = logging.getLogger(cfg.evaluation.evaluation_name)
    logger.setLevel(logging.INFO)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    log_dir = os.path.join(
        get_original_cwd(),
        cfg.evaluation.evaluation_name + "_" + dt_string + ".log",
    )
    cfg.model_path = os.path.join(get_original_cwd(), dt_string + "tmp.pt")
    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(str(OmegaConf.to_yaml(cfg)))
    # For reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    print(cfg.report_path)
    # ----------------------------
    #
    #            Main
    #
    # ----------------------------

    # Load dataset
    X = np.load(cfg.data.X_path)
    Y = np.load(cfg.data.Y_path)
    P = np.load(cfg.data.PID_path)  # participant IDs

    sample_rate = cfg.data.sample_rate
    task_type = cfg.data.task_type
    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    else:
        my_device = "cpu"
    # Expected shape of downstream X and Y
    # X: T x (Sample Rate*Epoch len) x 3
    # Y: T,
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    if task_type == "classify":
        print("\nLabel distribution:")
        print(pd.Series(Y).value_counts())
    elif task_type == "regress":
        print("\nOutput distribution:")
        Y_qnt = pd.Series(Y).quantile((0, 0.25, 0.5, 0.75, 1))
        Y_qnt.index = ("min", "25th", "median", "75th", "max")
        print(Y_qnt)

    if cfg.evaluation.feat_hand_crafted:
        print(
            """\n
        ##############################################
                    Hand-crafted features+RF
        ##############################################
        """
        )

        # Extract hand-crafted features
        print("Extracting features...")
        X_handfeats = pd.DataFrame(
            [handcraft_features(x, sample_rate=sample_rate) for x in tqdm(X)]
        )
        print("X_handfeats shape:", X_handfeats.shape)

        print("Train-test RF...")
        evaluate_feats(
            X_handfeats, Y, cfg, logger, groups=P, task_type=task_type
        )

    if cfg.evaluation.feat_random_cnn:
        print(
            """\n
        ##############################################
                    Random CNN features+RF
        ##############################################
        """
        )
        # Extract CNN features
        print("Extracting features...")
        if cfg.evaluation.network == "vgg":
            model = cnn1()
        else:
            # get cnn
            model = Resnet(output_size=cfg.data.output_size, cfg=cfg)
        model.to(my_device, dtype=torch.float)
        input_size = cfg.evaluation.input_size

        X_deepfeats = forward_by_batches(model, X, input_size, my_device)
        print("X_deepfeats shape:", X_deepfeats.shape)

        print("Train-test RF...")
        evaluate_feats(X_deepfeats, Y, cfg, logger, groups=P)

    if cfg.evaluation.flip_net:
        print(
            """\n
        ##############################################
                    Flip_net+RF
        ##############################################
        """
        )
        # Extract CNN features
        print("Extracting features...")
        cnn = cnn1()
        cnn.to(my_device, dtype=torch.float)
        load_weights(cfg.evaluation.flip_net_path, cnn, my_device)
        input_size = cfg.evaluation.input_size

        X_deepfeats = forward_by_batches(cnn, X, input_size, my_device)
        print("X_deepfeats shape:", X_deepfeats.shape)

        print("Train-test RF...")
        evaluate_feats(X_deepfeats, Y, cfg, logger, groups=P)

    """
    Start of MLP classifier evaluation
    """

    if cfg.evaluation.flip_net_ft:
        print(
            """\n
        ##############################################
                    Flip_net+MLP
        ##############################################
        """
        )
        # Original X shape: (1861541, 1000, 3) for capture24
        print("Original X shape:", X.shape)

        input_size = cfg.evaluation.input_size
        if X.shape[1] == input_size:
            print("No need to downsample")
            X_downsampled = X
        else:
            X_downsampled = resize(X, input_size)
        X_downsampled = X_downsampled.astype(
            "f4"
        )  # PyTorch defaults to float32
        # channels first: (N,M,3) -> (N,3,M). PyTorch uses channel first format
        X_downsampled = np.transpose(X_downsampled, (0, 2, 1))
        print("X transformed shape:", X_downsampled.shape)

        print("Train-test Flip_net+MLP...")
        evaluate_mlp(X_downsampled, Y, cfg, my_device, logger, groups=P)


if __name__ == "__main__":
    main()
