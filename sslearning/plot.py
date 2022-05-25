from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import os

LABEL_COLORS = {
    0: "#f95a5d",
    1: "#fda354",
    2: "#8c9d43",
    3: "#1c93b7",
    4: "#887ea5",
    5: "#3a3547",
}


def merge_rows(sleep_df, current_label, is_pred=False):
    if is_pred:
        sleep_df = sleep_df[sleep_df["sleep_stage_pred"] == current_label]
    else:
        sleep_df = sleep_df[sleep_df["sleep_stage"] == current_label]
    startTimes = []
    endTimes = []
    currentStartTime = None
    preTime = None
    labels = []

    for index, row in sleep_df.iterrows():
        currentTime = row["time"]
        if preTime is None:
            currentStartTime = currentTime
        else:
            if currentTime - preTime > timedelta(seconds=30):
                startTimes.append(currentStartTime)
                endTimes.append(preTime + timedelta(seconds=30))
                currentStartTime = currentTime
                labels.append(current_label)
        preTime = currentTime

    startTimes.append(currentStartTime)
    endTimes.append(preTime + timedelta(seconds=30))
    labels.append(current_label)

    stage_blocks = {
        "start_time": startTimes,
        "end_time": endTimes,
        "label": labels,
    }
    stage_df = pd.DataFrame(stage_blocks)
    return stage_df


def parse_file_name(file_path):
    file_name = file_path.split("/")[-1]
    file_name = file_name[:-4]  # remove .csv extension
    subjectID = file_name.split("_")[-1]
    file_date = file_name.split("_")[0]

    first_day = datetime.strptime(file_date, "%d%m%Y")

    if subjectID == "942099" or subjectID == "687006":
        # labels start on the same day:
        second_day = first_day
    else:
        second_day = first_day + timedelta(days=1)

    first_day_str = first_day.strftime("%d%m%Y")
    second_day_str = second_day.strftime("%d%m%Y")
    return first_day_str, second_day_str, subjectID


def xDate2yDate(xDate):
    year = xDate[:4]
    month = xDate[5:7]
    date = xDate[-2:]
    return date + month + year


def xName2yName(x_name, label_root):
    full_file_name = x_name.split("/")[-1]
    full_file_name = full_file_name[:-7]
    subject_id = full_file_name.split("_")[0]
    date_str = full_file_name.split("_")[-1]
    yDate = xDate2yDate(date_str)

    return subject_id, os.path.join(
        label_root, yDate + "_" + subject_id + ".csv"
    )


def updateTimes(first_day_str, second_day_str, x_df, y_df):
    current_date = first_day_str

    newtimes = []
    isSecondDay = False
    preHour = -1
    for index, row in y_df.iterrows():
        hour = int(row["time"].split(":")[0])
        if isSecondDay is False and (
            (hour < preHour and hour != 12) or (hour == 12 and preHour < hour)
        ):
            isSecondDay = True
            current_date = second_day_str
        row_time = datetime.strptime(
            current_date + " " + row["time"], "%d%m%Y %I:%M:%S %p"
        )
        newtimes.append(row_time)
        preHour = hour
    y_df["time"] = newtimes
    x_df["time"] = x_df["time"].apply(lambda x: x[:-27])
    x_df["time"] = x_df["time"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    )
    return x_df, y_df


def visu_subject(
    subject2visu,
    X_df,
    y_df,
    y_pred,
    time_df,
    pid_df,
    age=-1,
    ahi=-1,
    win_len=1,
):
    # TODO: add AHI and Age info
    # INPUT df are actually NP arrays
    subject_filter = pid_df == subject2visu
    # subject_y_pred = y_pred[subject_filter]
    my_time = time_df[subject_filter]
    if np.sum(subject_filter) != len(y_pred):
        # for cnn_lstm visu only
        y_pred_df = y_pred[subject_filter]
    else:
        y_pred_df = y_pred
    y_df = y_df[subject_filter]

    X_df = X_df[subject_filter]  # N * 2700
    # X_df = X_df.reshape(-1, 3, 900*win_len)
    avg_x = X_df[:, 0, :]
    avg_y = X_df[:, 1, :]
    avg_z = X_df[:, 2, :]

    # extra for win remove for general?
    avg_x = avg_x[:, -900:]
    avg_y = avg_y[:, -900:]
    avg_z = avg_z[:, -900:]

    avg_x = np.mean(avg_x, axis=1)
    avg_y = np.mean(avg_y, axis=1)
    avg_z = np.mean(avg_z, axis=1)

    # needs to change plotting for raw actigrahm
    # also change the visu for plotting
    # plotting
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(18, 12))
    ax.grid(True)
    ax.set_title(
        "Participant %s Age %d AHI %f " % (str(subject2visu), age, ahi),
        fontsize=16,
        fontweight="bold",
    )
    # format x-axis
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    ax.tick_params(axis="x", which="major", labelsize=20)

    ax.tick_params(axis="y", which="major", labelsize=20)
    ax.set_ylabel("Mean acceleration (mg)", fontsize=24, fontweight="bold")
    # format plot area
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ytick_locs = [
        1.0,
        0,
        -1,
        -1.25,
        -1.75,
        -2.25,
        -2.75,
        -3.25,
        -4.25,
        -4.75,
        -5.25,
        -5.75,
        -6.25,
    ]
    ylabels = [
        1.0,
        0,
        -1.0,
        "awake",
        "N1",
        "N2",
        "N3",
        "REM",
        "Pred:\nawake",
        "N1",
        "N2",
        "N3",
        "REM",
    ]
    plt.yticks(ytick_locs, ylabels)

    ax.plot(my_time, avg_x, color="black", label="axis 1")
    ax.plot(my_time, avg_y, color="yellow", label="axis 2")
    ax.plot(my_time, avg_z, color="blue", label="axis 3")

    # OVERLAY LABELS
    legendPatches = []
    legendLabels = []
    label_height = 0.5

    label_locs = {0: -1.5, 1: -2, 2: -2.5, 3: -3, 4: -3.5}
    label_locs_pred = {0: -4.5, 1: -5, 2: -5.5, 3: -6, 4: -6.5}
    label_name = {0: "awake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    frames = {
        "time": my_time,
        "sleep_stage": y_df,
        "sleep_stage_pred": y_pred_df,
    }
    y_df = pd.DataFrame(frames)
    for label in sorted(y_df["sleep_stage"].unique()):
        if label not in LABEL_COLORS.keys():
            continue
        legendPatches += [
            patches.Patch(color=LABEL_COLORS[label], label=label)
        ]
        legendLabels += [label_name[label]]

        stage_blocks_df = merge_rows(y_df, label)
        for ix, row in stage_blocks_df.iterrows():
            start = mdates.date2num(pd.to_datetime(row["start_time"]))
            end = mdates.date2num(pd.to_datetime(row["end_time"]))
            duration = row["end_time"] - row["start_time"]
            if (
                duration.total_seconds() < 10 * 3600
            ):  # make sure less than 10hrs
                ax.add_patch(
                    Rectangle(
                        (start, label_locs[label]),
                        end - start,
                        label_height,
                        color=LABEL_COLORS[label],
                    )
                )

    for label in sorted(y_df["sleep_stage_pred"].unique()):
        if label not in LABEL_COLORS.keys():
            continue
        legendPatches += [
            patches.Patch(color=LABEL_COLORS[label], label=label)
        ]
        legendLabels += [label_name[label]]

        stage_blocks_df = merge_rows(y_df, label, is_pred=True)
        for ix, row in stage_blocks_df.iterrows():
            start = mdates.date2num(pd.to_datetime(row["start_time"]))
            end = mdates.date2num(pd.to_datetime(row["end_time"]))
            duration = row["end_time"] - row["start_time"]
            if (
                duration.total_seconds() < 10 * 3600
            ):  # make sure less than 10hrs
                ax.add_patch(
                    Rectangle(
                        (start, label_locs_pred[label]),
                        end - start,
                        label_height,
                        color=LABEL_COLORS[label],
                    )
                )

    plt.show()
