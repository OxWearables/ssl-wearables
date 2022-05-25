"""
Preprocess PAMAP2 data into 30s and 10s windows with a 15s and 5 sec overlap respectively

Raw data:
Range +- 16g
Sample rate 100Hz

https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring

9-fold held one subject out CV
One in test
One in Val
Usage:
    python pamap.py
"""

from scipy import stats as s
from scipy import constants
import os
from tqdm import tqdm
import numpy as np
import glob

def get_data_content(data_path):
    # read flash.dat to a list of lists
    datContent = [i.strip().split() for i in open(data_path).readlines()]
    datContent = np.array(datContent)

    label_idx = 1
    timestamp_idx = 0
    x_idx = 4
    y_idx = 5
    z_idx = 6
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
    offset = overlap*sample_rate
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
    transition_class = 0
    for i in range(labels.shape[0]):
        row = labels[i, :]
        if np.sum(row == transition_class) > 0.5 * sample_count_per_row:
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


def process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap):
    X = []
    y = []
    pid = []

    for file_path in tqdm(file_paths):
        subject_id = int(file_path.split('/')[-1][-7:-4])
        datContent = get_data_content(file_path)
        current_X, current_y = content2x_and_y(datContent, epoch_len=epoch_len, overlap=overlap)
        current_X, current_y = clean_up_label(current_X, current_y)
        ids = np.full(shape=len(current_y), fill_value=subject_id, dtype=np.int)
        if len(X) == 0:
            X = current_X
            y = current_y
            pid = ids
        else:
            X = np.concatenate([X, current_X])
            y = np.concatenate([y, current_y])
            pid = np.concatenate([pid, ids])


    y = y.flatten()
    X = X / constants.g # convert to unit of g
    clip_value = 3
    X = np.clip(X, -clip_value, clip_value)

    # Keep only 8 activities that everyone has
    y_filter = (y == 1) | (y == 2) | (y == 3) | (y == 4) | (y == 12) | (y == 13) | (y == 16) | (y == 17)
    X = X[y_filter]
    y = y[y_filter]
    pid = pid[y_filter]

    np.save(X_path, X)
    np.save(y_path, y)
    np.save(pid_path, pid)


def get_write_paths(data_root):
    X_path = os.path.join(data_root, 'X.npy')
    y_path = os.path.join(data_root, 'Y.npy')
    pid_path = os.path.join(data_root, 'pid.npy')
    return X_path, y_path, pid_path


def main():
    data_root = '/data/UKBB/PAMAP2/'

    data_path = data_root+ 'Protocol/'
    protocol_file_paths = glob.glob(data_path + "*.dat")
    data_path = data_root + 'Optional/'
    optional_file_paths = glob.glob(data_path + "*.dat")
    file_paths = protocol_file_paths + optional_file_paths

    print("Processing for 30sec window..")
    data_root = '/data/UKBB/pamap_100hz_w30_o15/'
    X_path, y_path, pid_path = get_write_paths(data_root)
    epoch_len = 30
    overlap = 15
    process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap)
    print("Saved X to ", X_path)
    print("Saved y to ", y_path)

    print("Processing for 10sec window..")
    data_root = '/data/UKBB/pamap_100hz_w10_o5/'
    X_path, y_path, pid_path = get_write_paths(data_root)
    epoch_len = 10
    overlap = 5
    process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap)
    print("Saved X to ", X_path)
    print("Saved y to ", y_path)


if __name__ == "__main__":
    main()
