import sys
import torch
import numpy as np
from transforms3d.axangles import axangle2mat  # for rotation
from sklearn.decomposition import PCA
import random
import math

sys.path.append("../semi-supervised")
n_labels = 10
cuda = torch.cuda.is_available()


def train_test_split(X, y, group, num_test, context_df=None, time_df=None):
    """
    Get num_test subjects out from the X to be saved as test
    The rest will be treated as train
    """
    subject_list = np.unique(group)
    test_subject = np.random.choice(subject_list, num_test, replace=False)
    print(test_subject)
    test_idx = np.isin(group, test_subject)
    train_idx = ~test_idx

    X_test = X[test_idx]
    y_test = y[test_idx]
    group_test = group[test_idx]

    X_trian = X[train_idx]
    y_train = y[train_idx]
    group_train = group[train_idx]

    if context_df is not None:
        context_test = context_df[test_idx]
        context_train = context_df[train_idx]
    else:
        context_train = None
        context_test = None

    if time_df is None:
        return (
            X_trian,
            X_test,
            y_train,
            y_test,
            group_train,
            group_test,
            context_train,
            context_test,
        )
    else:
        time_test = time_df[test_idx]
        time_train = time_df[train_idx]
        return (
            X_trian,
            X_test,
            y_train,
            y_test,
            group_train,
            group_test,
            time_train,
            time_test,
            context_train,
            context_test,
        )


# https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
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


class Permutation_TimeSeries(object):
    """
    Rotation along an axis
    """

    def __call__(self, sample):
        # print("sampel shape")
        # print(sample.shape)
        # TIME_STEP * 3 * FEATURE_SIZE
        sample = np.swapaxes(sample, 1, 2)
        # MIN one segment
        sample = np.array(
            [
                DA_Permutation(
                    xi, nPerm=max(math.ceil(np.random.normal(2, 5)), 1)
                )
                for xi in sample
            ]
        )

        sample = np.swapaxes(sample, 1, 2)
        sample = torch.tensor(sample)
        return sample


class RotationAxis(object):
    """
    Rotation along an axis
    """

    def __call__(self, sample):
        # 3 * FEATURE_SIZE
        sample = np.swapaxes(sample, 0, 1)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        sample = np.matmul(sample, axangle2mat(axis, angle))
        sample = np.swapaxes(sample, 0, 1)
        return sample


class RotationAxisTimeSeries(object):
    """
    Every sample belongs to one subject
    Rotation along an axis
    """

    def __call__(self, sample):
        # print("sampel shape")
        # print(sample.shape)
        # TIME_STEP * 3 * FEATURE_SIZE
        axis = np.random.uniform(low=-1, high=1, size=sample.shape[1])
        angle = np.random.uniform(low=-np.pi, high=np.pi)

        sample = np.swapaxes(sample, 1, 2)
        sample = np.matmul(sample, axangle2mat(axis, angle))

        sample = np.swapaxes(sample, 1, 2)
        # sample = torch.tensor(sample)
        return sample


class RandomSwitchAxisTimeSeries(object):
    """
    Randomly switch the three axises for the raw files
    """

    def __call__(self, sample):
        # print(sample.shape)
        # TIME_STEP * 3 * FEATURE_SIZE
        x = sample[:, 0, :]
        y = sample[:, 1, :]
        z = sample[:, 2, :]

        choice = random.randint(1, 6)
        if choice == 1:
            sample = torch.stack([x, y, z], dim=1)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=1)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=1)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=1)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=1)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=1)
        # print(sample.shape)
        return sample


# class RandomSwitchAxisTimeSeries(object):
#     """
#     Randomly switch the three axises for the raw files
#     """
#
#     def __call__(self, sample):
#         # print(sample.shape)
#         # TIME_STEP * 3 * FEATURE_SIZE
#         x = sample[:, 0, :]
#         y = sample[:, 1, :]
#         z = sample[:, 2, :]
#
#         choice = torch.randint(1, 6, (1,))[0]
#         if choice == 1:
#             sample = torch.stack([x, y, z], dim=1)
#         elif choice == 2:
#             sample = torch.stack([x, z, y], dim=1)
#         elif choice == 3:
#             sample = torch.stack([y, x, z], dim=1)
#         elif choice == 4:
#             sample = torch.stack([y, z, x], dim=1)
#         elif choice == 5:
#             sample = torch.stack([z, x, y], dim=1)
#         elif choice == 6:
#             sample = torch.stack([z, y, x], dim=1)
#         # print(sample.shape)
#         return sample


class RandomSwitchAxis(object):
    """
    Randomly switch the three axises for the raw files
    Input size: 3 * FEATURE_SIZE
    """

    def __call__(self, sample):
        # print(sample.shape)
        # 3 * FEATURE
        x = sample[0, :]
        y = sample[1, :]
        z = sample[2, :]

        choice = random.randint(1, 6)

        if choice == 1:
            sample = torch.stack([x, y, z], dim=0)
        elif choice == 2:
            sample = torch.stack([x, z, y], dim=0)
        elif choice == 3:
            sample = torch.stack([y, x, z], dim=0)
        elif choice == 4:
            sample = torch.stack([y, z, x], dim=0)
        elif choice == 5:
            sample = torch.stack([z, x, y], dim=0)
        elif choice == 6:
            sample = torch.stack([z, y, x], dim=0)
        # print(sample.shape)
        return sample


# -------------------------------------------------------------------------------#
#                       AC added functions (18-09-21)
# -------------------------------------------------------------------------------#


"""DATA TRANSFORMATION OBJECT"""


class Transform(object):
    def __init__(
        self, transformations=None, channel_wise=False, limits=[-1, 1]
    ):
        self.transformations = transformations
        self.limits = np.asarray(limits)
        self.channel_wise = channel_wise

    def __call__(self, X):
        for aug_i in range(len(self.transformations)):
            if (
                self.transformations[aug_i]
                == "orientation_independent_transformation"
            ):
                X = orientation_independent_transformation(X)
            if self.transformations[aug_i] == "zscore":
                X = zscore_data(X, self.channel_wise)
            if self.transformations[aug_i] == "rescale":
                X = rescale_data(
                    X, self.channel_wise, self.limits[0], self.limits[1]
                )
        return X


# ----------------------------------------------------------------------------#
"""DATA TRANSFORMATION FUNCTIONS"""


def rescale_data(x, channel_wise=False, a=-1, b=1):
    """function to rescale the data"""
    # rescale data channel wise, versus over all channels
    if channel_wise is True:
        x_ = x.copy()
        print("transform shape: ", x.shape)
        for i in range(x.shape[1]):
            x_[:, i] = (
                np.multiply(
                    (b - a),
                    (x_[:, i] - np.nanmin(x_[:, i]))
                    / (np.nanmax(x_[:, i]) - np.nanmin(x_[:, i])),
                )
                + a
            )
        x = x_
    else:
        sz = x.shape
        x = np.reshape(x, (-1, 1))
        x = (
            np.multiply(
                (b - a), (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
            )
            + a
        )
        x = np.reshape(x, sz)
    return x


def zscore_data(x, channel_wise=False):
    """function to standardise the data (zscore)"""
    # zscore data channel wise, versus over all channels
    if channel_wise is True:
        x_ = x.copy()
        for i in range(x.shape[1]):
            x_[:, i] = (x_[:, i] - np.mean(x_[:, i])) / np.std(x_[:, i])
        x = x_
    else:
        sz = x.shape
        x = np.reshape(x, (-1, 1))
        x = (x - np.mean(x)) / np.std(x)
        x = np.reshape(x, sz)
    return x


def orientation_independent_transformation(data):
    # Function that takes in trixial inertial sensor data and translates to
    # a new orientation invarient coordinate system using three orthogonal
    # versors, whose orientation is independent of that of the smartphone and
    # alligned with gravity and the direction of motion :
    # Where:
    #   X: medio_lateral: phi,points forward, alligned with the
    #                     direction of motion
    #   Y: longitudinal: zeta, points upwards & parallel to users torso
    #   Z: anterior_posterior: xi, tracks lateral motion and orthogonal to
    #                     other two axis.
    # [1]    Gadaleta, M. and M. Rossi (2018). "IDNet: Smartphone-based gait
    #       recognition with convolutional neural
    #       networks." Pattern Recognition
    #       74(Supplement C): 25-37.

    n, nchan = data.shape

    # get accel. vectors
    aX = data[:, 0].reshape(n, 1)
    aY = data[:, 1].reshape(n, 1)
    aZ = data[:, 2].reshape(n, 1)

    # gravity versor
    pk = np.vstack((np.mean(aX), np.mean(aY), np.mean(aZ)))
    zeta = pk / np.linalg.norm(pk)
    A = np.concatenate((aX, aY, aZ), axis=1).T
    a_zeta = np.matmul(A.T, zeta)
    Af = A - (zeta * a_zeta.T)

    pca = PCA(n_components=3)
    pca.fit(Af.T)
    eigenvectors = pca.components_
    max_eigenvector = eigenvectors[:, 1]
    xi = max_eigenvector / np.linalg.norm(max_eigenvector)
    xi = xi.reshape(xi.shape[0], 1)

    a_xi = np.matmul(A.T, xi)
    phi = np.cross(zeta.reshape(-1), xi.reshape(-1))
    a_phi = np.matmul(A.T, phi).reshape(n, 1)

    transformed_data = np.concatenate((a_phi, a_zeta, a_xi), axis=1)

    return transformed_data
