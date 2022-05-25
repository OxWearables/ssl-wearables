"""
Preprocess Opportunity data into 30s and 10s windows with a 15s and 5 sec overlap respectively
We use +- 16g format
Sample Rate: 30 Hz
Unit: milli g
https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition

For a typical challenge, 1 sec sliding window and 50% overlap is used. Either specific sets or runs are used
for training or sometimes both run count and subject count are specified.



Usage:
    python oppo.py

"""

from scipy import stats as s
import numpy as np
import glob
import os
from tqdm import tqdm

def get_data_content(data_path):
    # read flash.dat to a list of lists
    datContent = [i.strip().split() for i in open(data_path).readlines()]
    datContent = np.array(datContent)

    label_idx = 243
    timestamp_idx = 0
    x_idx = 22
    y_idx = 23
    z_idx = 24
    index_to_keep = [timestamp_idx, label_idx, x_idx, y_idx, z_idx]
    # 3d +- 16 g

    datContent = datContent[:, index_to_keep]
    datContent = datContent.astype(float)
    datContent = datContent[~np.isnan(datContent).any(axis=1)]
    return datContent


def content2x_and_y(data_content, epoch_len=30, sample_rate=100, overlap=15):
    sample_count = int(np.floor(len(data_content) / (epoch_len * sample_rate)))

    sample_time_idx = 0
    sample_label_idx = 1
    sample_x_idx = 2
    sample_y_idx = 3
    sample_z_idx = 4

    sample_limit = sample_count * epoch_len * sample_rate
    data_content = data_content[:sample_limit, :]

    times = data_content[:, sample_time_idx]
    label = data_content[:, sample_label_idx]
    x = data_content[:, sample_x_idx]
    y = data_content[:, sample_y_idx]
    z = data_content[:, sample_z_idx]

    # to make overlappting window
    offset = overlap * sample_rate
    shifted_label = data_content[offset:-offset, sample_label_idx]
    shifted_x = data_content[offset:-offset:, sample_x_idx]
    shifted_y = data_content[offset:-offset:, sample_y_idx]
    shifted_z = data_content[offset:-offset:, sample_z_idx]

    shifted_label = shifted_label.reshape(-1, epoch_len * sample_rate)
    shifted_x = shifted_x.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y = shifted_y.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z = shifted_z.reshape(-1, epoch_len * sample_rate, 1)
    shifted_X = np.concatenate([shifted_x, shifted_y, shifted_z], axis=2)

    label = label.reshape(-1, epoch_len * sample_rate)
    x = x.reshape(-1, epoch_len * sample_rate, 1)
    y = y.reshape(-1, epoch_len * sample_rate, 1)
    z = z.reshape(-1, epoch_len * sample_rate, 1)
    X = np.concatenate([x, y, z], axis=2)

    X = np.concatenate([X, shifted_X])
    label = np.concatenate([label, shifted_label])
    return X, label


def clean_up_label(X, labels):
    # 1. remove rows with >50% zeros
    sample_count_per_row = labels.shape[1]

    rows2keep = np.ones(labels.shape[0], dtype=bool)
    for i in range(labels.shape[0]):
        row = labels[i, :]
        if np.sum(row == 0) > 0.5 * sample_count_per_row:
            rows2keep[i] = False

    labels = labels[rows2keep]
    X = X[rows2keep]

    # 2. majority voting for label in each epoch
    final_labels = []
    for i in range(labels.shape[0]):
        row = labels[i, :]
        final_labels.append(s.mode(row)[0])
    final_labels = np.array(final_labels, dtype=int)
    #print("Clean X shape: ", X.shape)
    #print("Clean y shape: ", final_labels.shape)
    return X, final_labels


def post_process_oppo(X, y, pid):
    zero_filter = np.array(y != 0)

    X = X[zero_filter]
    y = y[zero_filter]
    pid = pid[zero_filter]

    # change lie label from 5 to 3
    y[y == 5] = 3
    return X, y, pid


def process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap):
    X = []
    y = []
    pid = []
    sample_rate = 33

    for file_path in tqdm(file_paths):
        # print(file_path)
        subject_id = int(file_path.split('/')[-1][1:2])

        datContent = get_data_content(file_path)
        current_X, current_y = content2x_and_y(datContent, sample_rate=sample_rate,
                                               epoch_len=epoch_len, overlap=overlap)
        current_X, current_y = clean_up_label(current_X, current_y)
        if len(current_y) == 0:
            continue
        ids = np.full(shape=len(current_y), fill_value=subject_id, dtype=np.int)
        if len(X) == 0:
            X = current_X
            y = current_y
            pid = ids
        else:
            X = np.concatenate([X, current_X])
            y = np.concatenate([y, current_y])
            pid = np.concatenate([pid, ids])

    # post-process
    y = y.flatten()
    X = X / 1000 # convert to g
    clip_value = 3
    X = np.clip(X, -clip_value, clip_value)
    X, y, pid = post_process_oppo(X, y, pid)
    np.save(X_path, X)
    np.save(y_path, y)
    np.save(pid_path, pid)


def get_write_paths(data_root):
    X_path = os.path.join(data_root, 'X.npy')
    y_path = os.path.join(data_root, 'Y.npy')
    pid_path = os.path.join(data_root, 'pid.npy')
    return X_path, y_path, pid_path


def main():
    data_root = '/data/UKBB/opportunity/'
    data_path = data_root + 'dataset/'
    file_paths = glob.glob(data_path + "*.dat")

    print("Processing for 30sec window..")
    data_root = '/data/UKBB/oppo_33hz_w30_o15/'
    X_path, y_path, pid_path = get_write_paths(data_root)
    epoch_len = 30
    overlap = 15
    process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap)
    print("Saved X to ", X_path)
    print("Saved y to ", y_path)

    print("Processing for 10sec window..")
    data_root = '/data/UKBB/oppo_33hz_w10_o5/'
    X_path, y_path, pid_path = get_write_paths(data_root)
    epoch_len = 10
    overlap = 5
    process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap)
    print("Saved X to ", X_path)
    print("Saved y to ", y_path)


if __name__ == "__main__":
    main()
