"""
Esliger, D. W., Rowlands, A. V., Hurst, T. L., Catt, M., Murray, P., & Eston, R. G. (2011). Validation of the GENEA Accelerometer.
Chicago

Please contact Alex Rowlands to request the use of this dataset.
https://scholar.google.com.au/citations?user=o5n3S5kAAAAJ&hl=en
"""

import re
import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

DEVICE_HZ = 80  # Hz
# WINDOW_SEC = 30  # seconds
WINDOW_SEC = 10  # seconds
# WINDOW_OVERLAP_SEC = 15  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
DATAFILES = '/path/to/rowlands/data/P*_06[4,6].csv'  # only wrist!
# OUTDIR = '/path/to/rowlands_80hz_w30_o15'
OUTDIR = '/path/to/rowlands_80hz_w10_o5'


def is_good_quality(w):
    ''' Window quality check '''

    if w.isna().any().all():
        return False

    if len(w) != WINDOW_LEN:
        return False

    w_start, w_end = w.index[0], w.index[-1]
    w_duration = w_end - w_start
    target_duration = pd.Timedelta(WINDOW_SEC, 's')
    if np.abs(w_duration - target_duration) > WINDOW_TOL * target_duration:
        return False

    return True


def parse_filename(name):
    WEAR_CODE = {'035':'waist', '064':'left wrist', '066':'right wrist'}
    pattern = re.compile(r'(P\d{2})_(\d{3})', flags=re.IGNORECASE)  # P01_35
    m = pattern.search(os.path.basename(name))
    part, wear = m.group(1), m.group(2)
    wear = WEAR_CODE[wear]
    return part, wear


X, Y, T, P, W = [], [], [], [], []
for datafile in tqdm(glob.glob(DATAFILES)):
    data = pd.read_csv(datafile, parse_dates=['time'], index_col='time')

    part, wear = parse_filename(datafile)

    # Resample data
    period = int(round((1/DEVICE_HZ)*1000_000_000))
    data.resample(f'{period}N', origin='start').nearest(limit=1)

    for i in range(0, len(data), WINDOW_STEP_LEN):
        w = data.iloc[i:i+WINDOW_LEN]

        if not is_good_quality(w):
            # tqdm.write("Removed window with issue:")
            # tqdm.write(f"File: {datafile}")
            # tqdm.write(f"Start time: {w.index[0].strftime('%Y-%m-%d %H:%M:%S.%f')}")
            # tqdm.write(f"End time: {w.index[-1].strftime('%Y-%m-%d %H:%M:%S.%f')}")
            continue

        t = w.index[0].to_datetime64()
        x = w[['x','y','z']].values
        y = w['label'][0]

        # Combine these labels as they're extremely rare
        if (y=='Free-Living 10km/hr Run') or (y=='10km/hr Run') or (y=='12km/hr Run'):
            y = '10+km/hr Run'
        # Combine these too
        if (y=='Free-Living 6km/hr Walk'):
            y ='6km/hr Walk'

        X.append(x)
        Y.append(y)
        T.append(t)
        P.append(part)
        W.append(wear)

X = np.asarray(X)
Y = np.asarray(Y)
T = np.asarray(T)
P = np.asarray(P)
W = np.asarray(W)

os.system(f'mkdir -p {OUTDIR}')
np.save(os.path.join(OUTDIR, 'X'), X)
np.save(os.path.join(OUTDIR, 'Y'), Y)
np.save(os.path.join(OUTDIR, 'time'), T)
np.save(os.path.join(OUTDIR, 'pid'), P)
np.save(os.path.join(OUTDIR, 'wear'), W)

print(f"Saved in {OUTDIR}")
print("X shape:", X.shape)
print("Y distribution:")
print(pd.Series(Y).value_counts())
