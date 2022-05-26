"""
first run realworld_raw_preprocess.py
"""
import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .utils import resize

DEVICE_HZ = 50  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)
DATAFILES = "/Users/catong/repos/video-imu/data/realworld/imu/*.npy"
OUTDIR = "realworld_30hz_w10/"
LABEL_NAMES = [
    "jumping",
    "climbingup",
    "climbingdown",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]
BODY_PARTS = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]
start_ind = BODY_PARTS.index("forearm") * 3


def is_numpy_array_good_quality(window):
    """Window quality check"""

    if np.isnan(window).any():
        return False

    if len(window) != WINDOW_LEN:
        return False

    return True


X, Y, S, P = [], [], [], []
for datafile in tqdm(glob.glob(DATAFILES)):
    pid, sess_id, class_name, _ = datafile.split("/")[-1].split(".")
    data = np.load(datafile)[
        :, start_ind : start_ind + 3
    ]  # data corresponding to forearm

    for i in range(0, len(data), WINDOW_STEP_LEN):
        window = data[i : i + WINDOW_LEN]
        if not is_numpy_array_good_quality(window):
            continue
        X.append(window)
        Y.append(class_name)
        S.append(sess_id)
        P.append(pid)

X = np.asarray(X)
Y = np.asarray(Y)
S = np.asarray(S)
P = np.asarray(P)

# fixing unit to g
X = X / 9.81
# downsample to 30 Hz
X = resize(X, TARGET_WINDOW_LEN)

os.system(f"mkdir -p {OUTDIR}")
np.save(os.path.join(OUTDIR, "X"), X)
np.save(os.path.join(OUTDIR, "Y"), Y)
np.save(os.path.join(OUTDIR, "session_id"), S)
np.save(os.path.join(OUTDIR, "pid"), P)

print(f"Saved in {OUTDIR}")
print("X shape:", X.shape)
print("Y distribution:", len(set(Y)))
print(pd.Series(Y).value_counts())
print("User distribution:", len(set(P)))
print(pd.Series(P).value_counts())
