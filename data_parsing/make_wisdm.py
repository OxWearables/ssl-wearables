import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .utils import resize

DEVICE_HZ = 20  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)
DATAFILES = "/Users/catong/repos/video-imu/data/"
DATAFILES = DATAFILES + "wisdm/wisdm-dataset/raw/watch/accel/*.txt"
OUTDIR = "wisdm_30hz_w10/"


label_dict = {}
label_dict["walking"] = "A"
label_dict["jogging"] = "B"
label_dict["stairs"] = "C"
label_dict["sitting"] = "D"
label_dict["standing"] = "E"
label_dict["typing"] = "F"
label_dict["teeth"] = "G"
label_dict["soup"] = "H"
label_dict["chips"] = "I"
label_dict["pasta"] = "J"
label_dict["drinking"] = "K"
label_dict["sandwich"] = "L"
label_dict["kicking"] = "M"
label_dict["catch"] = "O"
label_dict["dribbling"] = "P"
label_dict["writing"] = "Q"
label_dict["clapping"] = "R"
label_dict["folding"] = "S"
code2name = {code: name for name, code in label_dict.items()}


def is_good_quality(w):
    """Window quality check"""

    if w.isna().any().any():
        return False

    if len(w) != WINDOW_LEN:
        return False

    # if len(w['annotation'].unique()) > 1:
    # return False

    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(WINDOW_SEC, "s")
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
        return False

    return True


# annolabel = pd.read_csv(ANNOLABELFILE, index_col='annotation')

X, Y, T, P, = (
    [],
    [],
    [],
    [],
)


def tmp(my_x):
    return float(my_x.strip(";"))


column_names = ["pid", "code", "time", "x", "y", "z"]

for datafile in tqdm(glob.glob(DATAFILES)):
    columns = ["pid", "class_code", "time", "x", "y", "z"]
    one_person_data = pd.read_csv(
        datafile,
        sep=",",
        header=None,
        converters={5: tmp},
        names=column_names,
        parse_dates=["time"],
        index_col="time",
    )
    one_person_data.index = pd.to_datetime(one_person_data.index)
    period = int(round((1 / DEVICE_HZ) * 1000_000_000))
    # one_person_data.resample(f'{period}N', origin='start').nearest(limit=1)
    code_to_df = dict(tuple(one_person_data.groupby("code")))
    pid = int(one_person_data["pid"][0])

    for code, data in code_to_df.items():
        try:
            data = data.resample(f"{period}N", origin="start").nearest(limit=1)
        except ValueError:
            if pid == 1629:
                data = data.drop_duplicates()
                data = data.resample(f"{period}N", origin="start").nearest(
                    limit=1
                )
                pass

        for i in range(0, len(data), WINDOW_STEP_LEN):
            w = data.iloc[i : i + WINDOW_LEN]

            if not is_good_quality(w):
                continue

            x = w[["x", "y", "z"]].values
            t = w.index[0].to_datetime64()

            X.append(x)
            Y.append(code2name[code])
            T.append(t)
            P.append(pid)

X = np.asarray(X)
Y = np.asarray(Y)
T = np.asarray(T)
P = np.asarray(P)

# fixing unit to g
X = X / 9.81
# downsample to 30 Hz
X = resize(X, TARGET_WINDOW_LEN)


os.system(f"mkdir -p {OUTDIR}")
np.save(os.path.join(OUTDIR, "X"), X)
np.save(os.path.join(OUTDIR, "Y"), Y)
np.save(os.path.join(OUTDIR, "time"), T)
np.save(os.path.join(OUTDIR, "pid"), P)

print(f"Saved in {OUTDIR}")
print("X shape:", X.shape)
print("Y distribution:", len(set(Y)))
print(pd.Series(Y).value_counts())
print("User distribution:", len(set(P)))
print(pd.Series(P).value_counts())
