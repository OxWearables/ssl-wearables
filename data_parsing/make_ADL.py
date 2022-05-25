import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import glob
from .utils import resize

DEVICE_HZ = 32  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 5  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30 # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)
OUTDIR = 'adl_30hz_w10/'

LABEL_NAMES = ['brush_teeth', 'climb_stairs', 'comb_hair', 'descend_stairs', 'drink_glass', 'eat_meat', 'eat_soup', 'getup_bed', 'liedown_bed', 'pour_water', 'sitdown_chair', 'standup_chair', 'use_telephone', 'walk']

FOLDER = '/Users/catong/repos/video-imu/data/HMP_Dataset/'
DATAFILES = '/Users/catong/repos/video-imu/data/HMP_Dataset/{}/*.txt'

def is_good_quality(w):
    ''' Window quality check '''

    if w.isna().any().any():
        return False

    if len(w) != WINDOW_LEN:
        return False

    return True

def main():
    X, Y, P = [], [], []
    for folder_name in os.listdir(FOLDER):
        print(folder_name)
        if folder_name.lower() in LABEL_NAMES or folder_name.endswith('MODEL'):
            print(folder_name)
            label_name = folder_name[:-6] if folder_name.endswith('MODEL') else folder_name
            label_name = label_name.lower()
            for filename in tqdm(glob.glob(DATAFILES.format(folder_name))):
                pid = filename.split('.')[-2].split('-')[-1]
                data = pd.read_csv(filename, delimiter=' ', header=None)
                data = data / 63 * 3 - 1.5

                if len(data) >= WINDOW_LEN:
                    for i in range(0, len(data), WINDOW_STEP_LEN):
                        window = data.iloc[i:i+WINDOW_LEN]
                        if not is_good_quality(window):
                            continue
                        X.append(window)
                        Y.append(label_name)
                        P.append(pid)
                else:
                    print('discarded!', folder_name, filename, len(data))


    X = np.asarray(X)
    Y = np.asarray(Y)
    P = np.asarray(P)
    X = resize(X, TARGET_WINDOW_LEN)
    df = pd.DataFrame({'y': Y, 'pid': P})
    print(df.groupby('pid')['y'].unique())

    os.system(f'mkdir -p {OUTDIR}')
    np.save(os.path.join(OUTDIR, 'X'), X)
    np.save(os.path.join(OUTDIR, 'Y'), Y)
    np.save(os.path.join(OUTDIR, 'pid'), P)

    print(f"Saved in {OUTDIR}")
    print("X shape:", X.shape)
    print("Y distribution:", len(set(Y)))
    print(pd.Series(Y).value_counts())
    print("User distribution:", len(set(P)))
    print(pd.Series(P).value_counts())
    import pdb; pdb.set_trace()

main()
