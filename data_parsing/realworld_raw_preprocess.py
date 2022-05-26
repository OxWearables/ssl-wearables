#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:07:15 2020

@author: yangao
"""

from zipfile import ZipFile
import os
import pandas as pd
import numpy as np

dataset_path = (
    "/Users/catong/repos/video-imu/data/realworld/realworld2016_dataset/"
)
save_path = "/Users/catong/repos/video-imu/data/realworld/imu/"
sub_ids_list = list(range(1, 16))
labels_list = [
    "climbingdown",
    "climbingup",
    "jumping",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]
body_parts_list = [
    "chest",
    "forearm",
    "head",
    "shin",
    "thigh",
    "upperarm",
    "waist",
]


def cal_min_time_len(readMe):
    time_len_list = []
    for line in readMe.readlines():
        if line.startswith(b"> entries"):
            time_len = int(line[11:16])
            time_len_list.append(time_len)

    min_time_len = min(time_len_list)
    print("min_time_len: ", min_time_len)
    return min_time_len


def cal_xyz(zip_file, csv, min_time_len):
    f = zip_file.open(csv)
    df = pd.read_csv(f)
    x = df["attr_x"]
    x = np.array(x)
    x = np.expand_dims(x, 1)
    y = df["attr_y"]
    y = np.array(y)
    y = np.expand_dims(y, 1)
    z = df["attr_z"]
    z = np.array(z)
    z = np.expand_dims(z, 1)
    xyz = np.concatenate((x, y, z), axis=1)
    xyz = xyz[:min_time_len, :]
    f.close()
    return xyz


def cal_xyz_acc(zip_file, i, sub_id, label):
    readme_file = zip_file.open("readMe")
    min_time_len = cal_min_time_len(readme_file)

    xyz_acc = None
    for b_part in body_parts_list:
        if i == 1:
            con1 = sub_id == 4 and label == "walking"
            con2 = sub_id == 6 and label == "sitting"
            con3 = sub_id == 7 and label == "sitting"
            con4 = sub_id == 8 and label == "standing"
            con5 = sub_id == 13 and label == "walking"
            if con1 or con2 or con3 or con4 or con5:
                csv_file_name = "acc_{}_2_{}.csv".format(label, b_part)
            else:
                csv_file_name = "acc_{}_{}.csv".format(label, b_part)
        else:
            csv_file_name = "acc_{}_{}_{}.csv".format(label, i, b_part)
        if csv_file_name in zip_file.namelist():
            xyz = cal_xyz(zip_file, csv_file_name, min_time_len)
            if b_part == "chest":
                xyz_acc = xyz
            else:
                xyz_acc = np.concatenate((xyz_acc, xyz), axis=1)
        else:
            print(
                "data missing: {}/{}; missing part: {}".format(
                    sub_id, label, b_part
                )
            )
            xyz = np.full([min_time_len, 3], np.nan)
            if b_part == "chest":
                xyz_acc = xyz
            else:
                xyz_acc = np.concatenate((xyz_acc, xyz), axis=1)
    return xyz_acc


def main_preprocess():
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for sub_id in sub_ids_list:
        imu_data_path = dataset_path + "proband{}/data/".format(sub_id)

        for label in labels_list:
            ziplabel = label.replace("_", "")
            zip_file_path = imu_data_path + "acc_{}_csv.zip".format(ziplabel)
            z_file = ZipFile(zip_file_path)
            if z_file.namelist()[0].endswith("zip"):
                save_zip_path = imu_data_path + "acc_{}_csv/".format(ziplabel)
                z_file.extractall(path=save_zip_path)

                for i in range(1, len(z_file.namelist()) + 1):
                    sub_z_file = ZipFile(
                        save_zip_path + "acc_{}_{}_csv.zip".format(ziplabel, i)
                    )
                    xyz_acc = cal_xyz_acc(sub_z_file, i, sub_id, ziplabel)
                    # save
                    file_name = "{}.{}.{}.npy".format(sub_id, i - 1, label)
                    np.save(save_path + file_name, xyz_acc)
                    sub_z_file.close()
                print(
                    "multiple: {}/{}; {} sessions in total".format(
                        sub_id, label, i
                    )
                )
            else:
                xyz_acc = cal_xyz_acc(z_file, 1, sub_id, ziplabel)
                # save
                file_name = "{}.0.{}.npy".format(sub_id, label)
                np.save(save_path + file_name, xyz_acc)
                print("{}/{}".format(sub_id, label))

            z_file.close()


main_preprocess()
