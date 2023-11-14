import os
import numpy as np
import pickle
import pandas as pd
import torch
import time

from sslearning.utils import trans30two1
import sslearning.myconstants as constants

# SSL
import sslearning.data.data_transformation as my_transforms
import glob
import torch.nn.functional as F


########################################
#
#   The end goal of this script should
#   provide three data loaders:
#   The train and test sets should contain distinct subjects
#   to allow for subject-wise cross-validation. The data
#   directory should have npy files for each subject to load everything
#   in memory.
#
#   1. train labeled   (x, y)
#   2. train unlabeled (x)
#   3. test labeled    (x)
#
#   This file also contains several different data classes that can be loaded
#   using different memory allocation and libraries.
#
########################################

# Return:
# x: batch_size * feature size (125)
# y: batch_size * label_size (5)

# START OF DATA CLASS
def convert_y_label(batch, label_pos):
    row_y = [item[1 + label_pos] for item in batch]
    master_y = torch.cat(row_y)
    final_y = master_y.long()
    return final_y


def subject_collate(batch):
    data = [item[0] for item in batch]
    data = torch.cat(data)

    aot_y = convert_y_label(batch, constants.TIME_REVERSAL_POS)
    scale_y = convert_y_label(batch, constants.SCALE_POS)
    permutation_y = convert_y_label(batch, constants.PERMUTATION_POS)
    time_w_y = convert_y_label(batch, constants.TIME_WARPED_POS)
    return [data, aot_y, scale_y, permutation_y, time_w_y]


def simclr_subject_collate(batch):
    x1 = [item[0] for item in batch]
    x1 = torch.cat(x1)
    x2 = [item[1] for item in batch]
    x2 = torch.cat(x2)
    return [x1, x2]


def worker_init_fn(worker_id):
    np.random.seed(int(time.time()))


def augment_view(X, cfg):
    new_X = []
    X = X.numpy()

    for i in range(len(X)):
        current_x = X[i, :, :]

        # choice = np.random.choice(
        #     2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
        # )[0]
        # current_x = my_transforms.flip(current_x, choice)
        # choice = np.random.choice(
        #     2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
        # )[0]
        # current_x = my_transforms.permute(current_x, choice)
        # choice = np.random.choice(
        #     2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
        # )[0]
        # current_x = my_transforms.time_warp(current_x, choice)
        choice = np.random.choice(
            2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
        )[0]
        current_x = my_transforms.rotation(current_x, choice)
        new_X.append(current_x)

    new_X = np.array(new_X)
    new_X = torch.Tensor(new_X)
    return new_X


def generate_labels(X, shuffle, cfg):
    labels = []
    new_X = []
    for i in range(len(X)):
        current_x = X[i, :, :]

        current_label = [0, 0, 0, 0]
        if cfg.task.time_reversal:
            choice = np.random.choice(
                2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
            )[0]
            current_x = my_transforms.flip(current_x, choice)
            current_label[constants.TIME_REVERSAL_POS] = choice

        if cfg.task.scale:
            choice = np.random.choice(
                2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
            )[0]
            current_x = my_transforms.scale(current_x, choice)
            current_label[constants.SCALE_POS] = choice

        if cfg.task.permutation:
            choice = np.random.choice(
                2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
            )[0]
            current_x = my_transforms.permute(current_x, choice)
            current_label[constants.PERMUTATION_POS] = choice

        if cfg.task.time_warped:
            choice = np.random.choice(
                2, 1, p=[cfg.task.positive_ratio, 1 - cfg.task.positive_ratio]
            )[0]
            current_x = my_transforms.time_warp(current_x, choice)
            current_label[constants.TIME_WARPED_POS] = choice

        new_X.append(current_x)
        labels.append(current_label)

    new_X = np.array(new_X)
    labels = np.array(labels)
    if shuffle:
        feature_size = new_X.shape[-1]
        new_X = np.concatenate([new_X, labels], axis=2)
        np.random.shuffle(new_X)

        labels = new_X[:, :, feature_size:]
        new_X = new_X[:, :, :feature_size]

    new_X = torch.Tensor(new_X)
    labels = torch.Tensor(labels)
    return new_X, labels


def check_file_list(file_list_path, data_root, cfg):
    if os.path.isfile(file_list_path) is False:
        csv_files = []

        if cfg.data.data_name == "100k":
            file_list = glob.glob(
                os.path.join(data_root, "*", "data", "*.npy")
            )
        else:
            file_list = glob.glob(data_root + "/*.npy")

        for file in file_list:
            csv_files.append(file)

        file_dict = {"file_list": csv_files}
        df = pd.DataFrame(file_dict)
        df.to_csv(file_list_path, index=False)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def separate_data(my_data):
    data = my_data[:, :, :30]
    data_std = my_data[:, :, -1][:, 0]

    return data, data_std


def time2window(sample, sample_len):
    """
    Convert time format of shape N x 3 x 30 data
    into an epoch format of 3 x sample_len
    Args:
        sample: data of shape N x 3 x 30
        sample_len (int):
            sample_rate x epoch_len (sec)
    Returns:
        final_sample (ny_array): shape 3 x sample_len
    """
    x = sample[:, 0, :]
    y = sample[:, 1, :]
    z = sample[:, 2, :]
    x = x.reshape(-1, sample_len)
    y = y.reshape(-1, sample_len)
    z = z.reshape(-1, sample_len)
    final_sample = np.concatenate((x, y, z), axis=0)
    return final_sample


def weighted_sample(
    data_with_std,
    num_sample=400,
    epoch_len=30,
    sample_rate=30,
    is_weighted_sample=False,
):
    """
    Weighted sample the windows that have most motion
    Args:
        data_with_std (np_array) of shape N x 3 x 31:
        last ele is the std per sec. We
        assume the sampling rate is 30hz.
        num_sample (int): windows to sample per subject
        epoch_len (int): how long should each epoch last in sec
        sample_rate (float): Sample frequency
        is_weighted_sample (boolean): random sampling if false
    Returns:
        sampled_data : high motion windows of size num_sample x 3 x 900
    """
    ori_data, data_std = separate_data(data_with_std)

    mov_avg = running_mean(data_std, epoch_len)
    if is_weighted_sample:
        sample_ides = np.random.choice(
            len(mov_avg),
            num_sample,
            replace=False,
            p=mov_avg / np.sum(mov_avg),
        )
    else:
        sample_ides = np.random.choice(len(mov_avg), num_sample, replace=False)

    sample_len = epoch_len * sample_rate
    channel_count = 3
    sampled_data = np.zeros([num_sample, channel_count, sample_len])
    for ii in range(num_sample):
        idx = sample_ides[ii]
        current_sample = ori_data[idx : idx + epoch_len, :, :]
        sampled_data[ii, :] = time2window(current_sample, sample_len)
    return sampled_data


def weighted_epoch_sample(data_with_std, num_sample=400):
    """
    Weighted sample the windows that have most motion
    Args:
        data_with_std (np_array) of shape N x 3 x 31:
         last ele is the std per sec. We
        assume the sampling rate is 30hz.
        num_sample (int): windows to sample per subject
        epoch_len (int): how long should each epoch last in sec
        sample_rate (float): Sample frequency
        is_weighted_sample (boolean): random sampling if false
    Returns:
        sampled_data : high motion windows of size num_sample x 3 x 900
    """
    ori_data = data_with_std[:, :, :300]
    data_std = data_with_std[:, :, -1][:, 0]

    sample_ides = np.random.choice(
        len(ori_data), num_sample, replace=False, p=data_std / np.sum(data_std)
    )

    sampled_data = np.zeros([num_sample, 3, 300])
    for ii in range(num_sample):
        idx = sample_ides[ii]
        # current_sample = ori_data[idx, :, :]
        sampled_data[ii, :] = ori_data[idx, :, :]
    return sampled_data


class SSL_dataset:
    def __init__(
        self,
        data_root,
        file_list_path,
        cfg,
        transform=None,
        shuffle=False,
        is_epoch_data=False,
    ):
        """
        Args:
            data_root (string): directory containing all data files
            file_list_path (string): file list
            cfg (dict): config
            shuffle (bool): whether permute epoches within one subject
            is_epoch_data (bool): whether each sample is one
            second of data or 10 seconds of data


        Returns:
            data : transformed sample
            labels (dict) : labels for avalaible transformations
        """
        check_file_list(file_list_path, data_root, cfg)
        file_list_df = pd.read_csv(file_list_path)
        self.file_list = file_list_df["file_list"].to_list()
        self.data_root = data_root
        self.cfg = cfg
        self.is_epoch_data = is_epoch_data
        self.ratio2keep = cfg.data.ratio2keep
        self.shuffle = shuffle
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # idx starts from zero
        file_to_load = self.file_list[idx]
        X = np.load(file_to_load, allow_pickle=True)

        # to help select a percentage of data per subject
        subject_data_count = int(len(X) * self.ratio2keep)
        assert subject_data_count >= self.cfg.dataloader.num_sample_per_subject
        if self.ratio2keep != 1:
            X = X[:subject_data_count, :]

        if self.is_epoch_data:
            X = weighted_epoch_sample(
                X, num_sample=self.cfg.dataloader.num_sample_per_subject
            )
        else:
            X = weighted_sample(
                X,
                num_sample=self.cfg.dataloader.num_sample_per_subject,
                epoch_len=self.cfg.dataloader.epoch_len,
                sample_rate=self.cfg.dataloader.sample_rate,
                is_weighted_sample=self.cfg.data.weighted_sample,
            )

        X, labels = generate_labels(X, self.shuffle, self.cfg)

        if self.transform:
            X = self.transform(X)

        return (
            X,
            labels[:, constants.TIME_REVERSAL_POS],
            labels[:, constants.SCALE_POS],
            labels[:, constants.PERMUTATION_POS],
            labels[:, constants.TIME_WARPED_POS],
        )


class SIMCLR_dataset:
    def __init__(
        self,
        data_root,
        file_list_path,
        cfg,
        transform=None,
        shuffle=False,
        is_epoch_data=False,
    ):
        """
        Args:
            data_root (string): directory containing all data files
            file_list_path (string): file list
            cfg (dict): config
            shuffle (bool): whether permute epoches within one subject
            is_epoch_data (bool): whether each sample is one
            second of data or 10 seconds of data


        Returns:
            data : transformed sample
            labels (dict) : labels for avalaible transformations
        """
        check_file_list(file_list_path, data_root, cfg)
        file_list_df = pd.read_csv(file_list_path)
        self.file_list = file_list_df["file_list"].to_list()
        self.data_root = data_root
        self.cfg = cfg
        self.is_epoch_data = is_epoch_data
        self.ratio2keep = cfg.data.ratio2keep
        self.shuffle = shuffle
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print(idx)

        # idx starts from zero
        file_to_load = self.file_list[idx]
        X = np.load(file_to_load, allow_pickle=True)

        # to help select a percentage of data per subject
        subject_data_count = int(len(X) * self.ratio2keep)
        assert subject_data_count >= self.cfg.dataloader.num_sample_per_subject
        if self.ratio2keep != 1:
            X = X[:subject_data_count, :]

        if self.is_epoch_data:
            X = weighted_epoch_sample(
                X, num_sample=self.cfg.dataloader.num_sample_per_subject
            )
        else:
            X = weighted_sample(
                X,
                num_sample=self.cfg.dataloader.num_sample_per_subject,
                epoch_len=self.cfg.dataloader.epoch_len,
                sample_rate=self.cfg.dataloader.sample_rate,
                is_weighted_sample=self.cfg.data.weighted_sample,
            )

        X = torch.from_numpy(X)
        if self.transform:
            X = self.transform(X)

        X1 = augment_view(X, self.cfg)
        X2 = augment_view(X, self.cfg)
        return (X1, X2)


# Return:
# x: batch_size * feature size (125)
# y: batch_size * label_size (5)
class RegularDataset:
    def __init__(
        self, data_path, file_list_path, epoch_count_path, transform=None
    ):
        """
        Args:
            data_path (string): path to data
            file_name_path (string): path to file list
        """
        self.file_list = pickle.load(open(file_list_path, "rb"))
        self.epoch_count = pickle.load(open(epoch_count_path, "rb"))
        self.epoch_cumsum = np.cumsum(self.epoch_count)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return self.epoch_cumsum[-1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # idx starts from zero
        file_id = np.searchsorted(self.epoch_cumsum, idx - 1)
        file_to_load = os.path.join(
            self.data_path, self.file_list[file_id] + ".npz"
        )
        X = np.load(file_to_load, mmap_mode="r", allow_pickle=True)
        if file_id == 0:
            row_id = idx - 1
        else:
            row_id = idx - self.epoch_cumsum[file_id] - 1
        sample = X[row_id, :]
        if self.transform:
            sample = self.transform(sample)

        return sample


class SlidingWindowDataset:
    def get_lookup_table(self, meta_df):
        ids_inplace = meta_df["pid"].unique()
        firstIndexes = []
        for my_id in ids_inplace:
            firstIndexes.append(
                meta_df["pid"]
                .where(meta_df["pid"] == my_id)
                .first_valid_index()
            )

        new_idx = 0
        lookupTable = {}
        # The lookup table will always store the index of the
        # epoch that we wish to classify in the full dataframe
        # In the getItem method, one only needs to load the
        # neighboring epcohes depending on the config.

        for i in range(len(firstIndexes)):
            # increment subject after subject
            currentStartingIdx = firstIndexes[i]

            # compute sequence length
            if i == len(firstIndexes) - 1:
                seq_length = len(meta_df["pid"]) - currentStartingIdx
            else:
                seq_length = firstIndexes[i + 1] - currentStartingIdx
            new_length = seq_length - self.win_length + 1

            # decide starting idx in the original df
            if self.isBidirectional:
                currentStartingIdx += int((self.win_length - 1) / 2)
            else:
                currentStartingIdx += self.win_length - 1

            for j in range(new_length):
                lookupTable[new_idx] = currentStartingIdx + j
                new_idx += 1
        return lookupTable

    def __init__(
        self,
        X,
        y=[],
        context_data=None,
        isLabel=False,
        win_length=1,
        isBidirectional=False,
        pid_list=None,
        transform=None,
        target_transform=None,
    ):
        """
        Y needs to be in one-hot encoding
        X needs to be in N * Width
        Args:
            data_path (string): path to data
            files_to_load (list): subject names
            currently all npz format should allow support multiple ext
            win_length: specify the epoch length of the sliding window
            isBidirectional: if true, the win_length has to
            be odd and the target epoch is the on is the middle.
             If false, the win_length needs to be >= 1,
            and the target epoch is the last one
            pid_list: a list or numpy array
        """

        if isBidirectional and win_length % 2 == 0:
            raise ValueError(
                "isBidirectional is True but win_length is not odd!"
            )

        pid_list = pd.DataFrame(data=pid_list, columns=["pid"])
        ids_inplace = pid_list["pid"].unique()
        lastIndexes = []
        for my_id in ids_inplace:
            lastIndexes.append(
                pid_list["pid"]
                .where(pid_list["pid"] == my_id)
                .last_valid_index()
            )

        self.win_length = win_length
        self.isBidirectional = isBidirectional
        self.lastIndexes = np.array(lastIndexes)
        self.X = torch.from_numpy(X)
        self.y = y
        if context_data is not None:
            self.context_data = torch.from_numpy(context_data)
        else:
            self.context_data = None
        self.isLabel = isLabel
        self.transform = transform
        self.targetTransform = target_transform
        print("Total sample count : " + str(len(self.X)))
        self.lookupTable = self.get_lookup_table(pid_list)

    def __len__(self):
        num_subject = len(self.lastIndexes)
        return len(self.X) - num_subject * (self.win_length - 1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx2load = self.lookupTable[idx]
        if self.isBidirectional:
            halfWinLength = int((self.win_length - 1) / 2)
            sample = self.X[
                idx2load - halfWinLength : idx2load + halfWinLength + 1, :
            ]  # the second idx is not inclusive
        else:
            sample = self.X[idx2load - self.win_length + 1 : idx2load + 1, :]

        if self.context_data is not None:
            context_data = self.context_data[idx2load, :]
            context_data = torch.flatten(context_data)
        sample = sample.permute(1, 0, 2)  # From N * C * F -> C * N *F
        sample = torch.flatten(sample, 1)

        y = []
        if self.isLabel:
            y = self.y[idx2load]

            if self.targetTransform:
                y = self.targetTransform(y)

        if self.transform:
            sample = self.transform(sample)
        if self.context_data is None:
            return sample, y
        else:
            return sample, y, context_data


class cnnLSTMDataset:
    def __init__(self, X, pid=[], y=[], transform=None, target_transform=None):
        """
        Y needs to be in one-hot encoding
        X needs to be in N * Width
        Pid is a numpy array of size N
        Args:
            data_path (string): path to data
            files_to_load (list): subject names
            currently all npz format should allow support multiple ext

        """

        self.X = torch.from_numpy(X)
        self.y = y
        self.pid = pid
        self.unique_pid_list = np.unique(pid)
        self.transform = transform
        self.targetTransform = target_transform
        print("Total sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.unique_pid_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid_of_choice = self.unique_pid_list[idx]
        sample_filter = self.pid == pid_of_choice
        sample = self.X[sample_filter, :]

        y = self.y[sample_filter]
        if self.targetTransform:
            y = [self.targetTransform(ele) for ele in y]
            # y = self.targetTransform(y)

        if self.transform:
            sample = self.transform(sample)

        return sample, y, self.pid[sample_filter]


class NormalDataset:
    def __init__(
        self,
        X,
        y=[],
        pid=[],
        name="",
        isLabel=False,
        transform=None,
        target_transform=None,
    ):
        """
        Y needs to be in one-hot encoding
        X needs to be in N * Width
        Args:
            data_path (string): path to data
            files_to_load (list): subject names
            currently all npz format should allow support multiple ext

        """

        self.X = torch.from_numpy(X)
        self.y = y
        self.isLabel = isLabel
        self.transform = transform
        self.targetTransform = target_transform
        self.pid = pid
        print(name + " set sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]
        y = []
        if self.isLabel:
            y = self.y[idx]
            if self.targetTransform:
                y = self.targetTransform(y)

        if self.transform:
            sample = self.transform(sample)
        if len(self.pid) >= 1:
            return sample, y, self.pid[idx]
        else:
            return sample, y


def generate_labels_double(X, shuffle):
    labels = []
    new_X = []
    for i in range(len(X)):
        current_x = X[i, :, :]
        # rotatioin: 0
        # axis_switch: 1

        # We will have the epoch label for all tasks even
        # if when we don't use the tasks
        # We only set the task specific label when we use it
        choice = 0
        trans_x = my_transforms.flip(current_x, choice)
        new_X.append(trans_x)
        labels.append([-1, -1, choice])

        choice = 1
        trans_x = my_transforms.flip(current_x, choice)
        new_X.append(trans_x)
        labels.append([-1, -1, choice])

    new_X = np.array(new_X)
    labels = np.array(labels)
    if shuffle:
        feature_size = new_X.shape[-1]
        new_X = np.concatenate([new_X, labels], axis=2)
        np.random.shuffle(new_X)

        labels = new_X[:, :, feature_size:]
        new_X = new_X[:, :, :feature_size]

    new_X = torch.Tensor(new_X)
    labels = torch.Tensor(labels)

    return new_X, labels


class subject_dataset:
    def __init__(self, data_root, num_sample_per_subject=1500, has_std=False):
        """
        Enumerates weighted sampled for a single subject

        Args:
            data_root (string): path to subject npy file

        Returns:
            data : transformed sample
            labels (dict) : labels for avalaible transformations
        """
        self.data_root = data_root

        data = np.load(data_root, allow_pickle=True)
        if has_std:
            X = weighted_sample(
                data, num_sample=num_sample_per_subject, epoch_len=10
            )

        else:
            # transform to N x 3 x 31
            master_data = []

            for i in range(len(data)):
                data_i = trans30two1(data[i])
                master_data.append(data_i)
            master_data = np.array(master_data)
            master_data = master_data.reshape(-1, 3, 30)

            # Compute the std for each example
            std_data = np.std(master_data, axis=2).reshape(-1, 3, 1)
            master_data = np.concatenate((master_data, std_data), axis=2)

            # Get all the wegihted samples
            X = weighted_sample(
                master_data, num_sample=num_sample_per_subject, epoch_len=10
            )

        X, labels = generate_labels_double(X, False)
        self.num_size = len(X)

        # index of the samples
        self.sample_index = np.repeat(np.arange(int(len(X) / 2)), 2)

        labels = labels[:, 2]  # only the last idx is time revseral

        self.X = X
        y = F.one_hot(labels.to(torch.int64), num_classes=2)
        self.y = y

    def __len__(self):
        return self.num_size

    def __getitem__(self, idx):
        X = self.X[idx, :, :]
        y = self.y[idx]

        return X, y
