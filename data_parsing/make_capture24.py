import re
import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DEVICE_HZ = 100  # Hz
WINDOW_SEC = 10  # seconds
# WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
DATAFILES = "capture24/data/*.csv"
ANNOLABELFILE = "capture24/annotation-label-dictionary.csv"
# OUTDIR = 'capture24_100hz_w10_o5/'
OUTDIR = "capture24_100hz_w10_o0/"


def is_good_quality(w):
    """Window quality check"""

    if w.isna().any().any():
        return False

    if len(w) != WINDOW_LEN:
        return False

    if len(w["annotation"].unique()) > 1:
        return False

    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(WINDOW_SEC, "s")
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
        return False

    return True


annolabel = pd.read_csv(ANNOLABELFILE, index_col="annotation")

X, Y, T, P, = (
    [],
    [],
    [],
    [],
)

for datafile in tqdm(glob.glob(DATAFILES)):
    data = pd.read_csv(
        datafile,
        parse_dates=["time"],
        index_col="time",
        dtype={"x": "f4", "y": "f4", "z": "f4", "annotation": "str"},
    )

    p = re.search(r"(P\d{3})", datafile, flags=re.IGNORECASE).group()

    for i in range(0, len(data), WINDOW_STEP_LEN):
        w = data.iloc[i : i + WINDOW_LEN]

        if not is_good_quality(w):
            continue

        t = w.index[0].to_datetime64()
        x = w[["x", "y", "z"]].values
        y = annolabel.loc[w["annotation"][0], "label:Willetts2018"]

        if y == "sleep":
            continue  # ignore sleep

        X.append(x)
        Y.append(y)
        T.append(t)
        P.append(p)

X = np.asarray(X)
Y = np.asarray(Y)
T = np.asarray(T)
P = np.asarray(P)

os.system(f"mkdir -p {OUTDIR}")
np.save(os.path.join(OUTDIR, "X"), X)
np.save(os.path.join(OUTDIR, "Y"), Y)
np.save(os.path.join(OUTDIR, "time"), T)
np.save(os.path.join(OUTDIR, "pid"), P)

print(f"Saved in {OUTDIR}")
print("X shape:", X.shape)
print("Y distribution:")
print(pd.Series(Y).value_counts())
