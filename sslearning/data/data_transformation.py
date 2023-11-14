import numpy as np
from transforms3d.axangles import axangle2mat  # for rotation
from scipy.interpolate import CubicSpline  # for warping

"""
This file implements a list of transforms for tri-axial raw-accelerometry
We assume that the input format is of size:
3 x (epoch_len * sampling_frequency)

Transformations included:
0. jitter
1. Rotation: degree
2. Channel shuffling: which axis is being switched
3. Horizontal flip: binary
4. Permutation: binary

This script is mostly based off from
https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.py
"""


def rotation(sample, choice):
    """
    Rotate along one axis

    Args:
        sample (numpy array):  3 * FEATURE_SIZE
        choice (float): [0, 9] for each axis,
        we can do 4 rotations 0, 90 180, 270
    """
    if choice == 1:
        # angle_choices = [1 / 4 * np.pi, 1 / 2 * np.pi, 3 / 4 * np.pi]
        # angle = angle_choices[choice % 3]
        # axis = axis_choices[math.floor(choice / 3)]

        axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        sample = np.swapaxes(sample, 0, 1)
        for i in range(3):
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            sample = np.matmul(sample, axangle2mat(axes[i], angle))

        sample = np.swapaxes(sample, 0, 1)
    return sample


def switch_axis(sample, choice):
    """
    Randomly switch the three axises for the raw files

    Args:
        sample (numpy array): 3 * FEATURE_SIZE
        choice (int): 0-6 for direction selection
    """
    x = sample[0, :]
    y = sample[1, :]
    z = sample[2, :]

    if choice == 0:
        return sample
    elif choice == 1:
        sample = np.stack([x, y, z], axis=0)
    elif choice == 2:
        sample = np.stack([x, z, y], axis=0)
    elif choice == 3:
        sample = np.stack([y, x, z], axis=0)
    elif choice == 4:
        sample = np.stack([y, z, x], axis=0)
    elif choice == 5:
        sample = np.stack([z, x, y], axis=0)
    elif choice == 6:
        sample = np.stack([z, y, x], axis=0)
    return sample


def flip(sample, choice):
    """
    Flip over the actigram on the temporal scale

    Args:
        sample (numpy array): 3 * FEATURE_SIZE
        choice (int): 0-1 binary
    """
    if choice == 1:
        sample = np.flip(sample, 1)
    return sample


def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile is True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(
            np.random.randint(
                minSegLength, X.shape[0] - minSegLength, nPerm - 1
            )
        )
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]] : segs[idx[ii] + 1], :]
        X_new[pp : pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return X_new


def permute(sample, choice, nPerm=4, minSegLength=10):
    """
    Distort an epoch by dividing up the sample into several segments and
    then permute them

    Args:
        sample (numpy array): 3 * FEATURE_SIZE
        choice (int): 0-1 binary
    """
    if choice == 1:
        sample = np.swapaxes(sample, 0, 1)
        sample = DA_Permutation(sample, nPerm=nPerm, minSegLength=minSegLength)
        sample = np.swapaxes(sample, 0, 1)
    return sample


def is_scaling_factor_invalid(scaling_factor, min_scale_sigma):
    """
    Ensure each of the abs values of the scaling
    factors are greater than the min
    """
    for i in range(len(scaling_factor)):
        if abs(scaling_factor[i] - 1) < min_scale_sigma:
            return True
    return False


def DA_Scaling(X, sigma=0.3, min_scale_sigma=0.05):
    scaling_factor = np.random.normal(
        loc=1.0, scale=sigma, size=(1, X.shape[1])
    )  # shape=(1,3)
    while is_scaling_factor_invalid(scaling_factor, min_scale_sigma):
        scaling_factor = np.random.normal(
            loc=1.0, scale=sigma, size=(1, X.shape[1])
        )
    my_noise = np.matmul(np.ones((X.shape[0], 1)), scaling_factor)
    X = X * my_noise
    return X


def scaling_uniform(X, scale_range=0.15, min_scale_diff=0.02):
    low = 1 - scale_range
    high = 1 + scale_range
    scaling_factor = np.random.uniform(
        low=low, high=high, size=(X.shape[1])
    )  # shape=(3)
    while is_scaling_factor_invalid(scaling_factor, min_scale_diff):
        scaling_factor = np.random.uniform(
            low=low, high=high, size=(X.shape[1])
        )

    for i in range(3):
        X[:, i] = X[:, i] * scaling_factor[i]

    return X


def scale(sample, choice, scale_range=0.5, min_scale_diff=0.15):
    if choice == 1:
        sample = np.swapaxes(sample, 0, 1)
        sample = scaling_uniform(
            sample, scale_range=scale_range, min_scale_diff=min_scale_diff
        )
        sample = np.swapaxes(sample, 0, 1)
    return sample


def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(
        X, sigma
    )  # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [
        (X.shape[0] - 1) / tt_cum[-1, 0],
        (X.shape[0] - 1) / tt_cum[-1, 1],
        (X.shape[0] - 1) / tt_cum[-1, 2],
    ]
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    return tt_cum


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (
        np.ones((X.shape[1], 1))
        * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))
    ).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()


def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
    X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])
    return X_new


def time_warp(sample, choice, sigma=0.2):
    if choice == 1:
        sample = np.swapaxes(sample, 0, 1)
        sample = DA_TimeWarp(sample, sigma=sigma)
        sample = np.swapaxes(sample, 0, 1)
    return sample
