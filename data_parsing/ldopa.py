import os
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from scipy import constants
from joblib import Parallel, delayed
import numpy as np
import warnings
import argparse
import actipy
from scipy.interpolate import interp1d
from pathlib import Path
from shutil import rmtree, copyfile
from glob import glob
import synapseclient
import synapseutils
from dotenv import load_dotenv, find_dotenv

RAW_DIR = "Data/LDOPA_DATA/"
PROCESSED_DIR = "Data/Ldopa_Processed/"
N_JOBS = 8
PROCESS_ARGS = {
    "load_data_args": {"sample_rate": 50, "annot_type": str},
    "make_windows_args": {"sample_rate": 50, "label_type": "mode"},
}
DATAFILES = "Ldopa_Processed/acc_data/*.csv"

LDOPA_DOWNLOADS = [
    ["UPDRSResponses", "syn20681939"],
    ["TaskScoresPartII", "syn20681938"],
    ["TaskScoresPartI", "syn20681937"],
    ["TaskCodeDictionary", "syn20681936"],
    ["SensorDataPartII", "syn20681932"],
    ["SensorDataPartI", "syn20681931"],
    ["MetadataOfPatientOnboardingDictionary", "syn20681895"],
    ["MetadataOfPatientOnboarding", "syn20681894"],
    ["MetadataOfLaboratoryVisits", "syn20681892"],
    ["HomeTasks", "syn20681035"],
]


def load_environment_vars(env_strings=[]):
    load_dotenv(find_dotenv())
    missing_envs = []
    env_values = []

    for env_string in env_strings:
        env_value = os.getenv(env_string)
        if env_value is None or env_value == "":
            missing_envs.append(env_string)
        else:
            env_values.append(env_value)

    if missing_envs:
        missing_envs_str = ", ".join(missing_envs)
        raise ValueError(
            f"Please set the following environment variable(s) in the .env file: {missing_envs_str}"
        )

    return tuple(env_values)


USERNAME, APIKEY = load_environment_vars(
    ["SYNAPSE_USERNAME", "SYNAPSE_APIKEY"]
)


def check_files_exist(dir, files):
    return all(os.path.exists(os.path.join(dir, file)) for file in files)


def get_first_file(dataFolder, folderName):
    return os.path.join(
        dataFolder,
        folderName,
        os.listdir(os.path.join(dataFolder, folderName))[0],
    )


def parse_datetime_from_timestamp(time_in_secs):
    return datetime.fromtimestamp(float(time_in_secs))


def build_metadata(datadir=RAW_DIR, processeddir=PROCESSED_DIR):
    os.makedirs(processeddir, exist_ok=True)
    outFile = os.path.join(processeddir, "Metadata.csv")

    if os.path.exists(outFile):
        return pd.read_csv(outFile, index_col=[0])

    metadata = pd.read_csv(
        os.path.join(datadir, "MetadataOfPatientOnboarding.csv"), index_col=[0]
    )

    updrs_cols = [
        "updrs_score_p1",
        "updrs_score_p2",
        "updrs_score_p3",
        "updrs_score_p4",
        "updrs_second_visit_score_p3",
    ]

    metadata["MeanUPDRS"] = metadata[updrs_cols].mean(axis=1)

    metadata = metadata[["gender", "MeanUPDRS"]]

    metadata.to_csv(outFile)

    return metadata


def build_acc_data(datadir=RAW_DIR, processeddir=PROCESSED_DIR, n_jobs=N_JOBS):
    subjects = build_task_reference_file(datadir, processeddir)[
        "subject_id"
    ].unique()

    outdir = os.path.join(processeddir, "acc_data")
    os.makedirs(outdir, exist_ok=True)

    if len(glob(os.path.join(outdir, "*.csv"))) != len(subjects):
        Parallel(n_jobs=n_jobs)(
            delayed(build_participant_acc_data)(subject, datadir, outdir)
            for subject in tqdm(subjects)
        )

    else:
        print("Acceleration data already compiled...\n")


def build_task_reference_file(
    datadir=RAW_DIR, outdir=PROCESSED_DIR, overwrite=False
):
    outFile = os.path.join(outdir, "TaskReferenceFile.csv")

    if os.path.exists(outFile) and not overwrite:
        taskRefFile = pd.read_csv(
            outFile, parse_dates=["timestamp_start", "timestamp_end"]
        )
        print(f'Using saved task reference file saved at "{outFile}".')

    else:
        os.makedirs(outdir, exist_ok=True)

        taskScoreFile1 = os.path.join(datadir, "TaskScoresPartI.csv")
        taskScoreFile2 = os.path.join(datadir, "TaskScoresPartII.csv")
        homeTaskFile = os.path.join(datadir, "HomeTasks.csv")

        taskScore1 = pd.read_csv(
            taskScoreFile1,
            parse_dates=["timestamp_start", "timestamp_end"],
            date_parser=parse_datetime_from_timestamp,
        )
        taskScore2 = pd.read_csv(
            taskScoreFile2,
            parse_dates=["timestamp_start", "timestamp_end"],
            date_parser=parse_datetime_from_timestamp,
        )
        taskScores = pd.concat([taskScore1, taskScore2])[
            [
                "subject_id",
                "visit",
                "task_code",
                "timestamp_start",
                "timestamp_end",
            ]
        ]
        visit_to_day = {1: 1, 2: 4}

        taskScores["participant_day"] = taskScores["visit"].map(visit_to_day)
        taskScores.drop("visit", axis=1, inplace=True)

        homeTasks = pd.read_csv(
            homeTaskFile,
            parse_dates=["timestamp_start", "timestamp_end"],
            date_parser=parse_datetime_from_timestamp,
        )
        homeTasks = homeTasks[
            [
                "subject_id",
                "participant_day",
                "task_code",
                "timestamp_start",
                "timestamp_end",
            ]
        ]

        taskRefFile = (
            pd.concat([taskScores, homeTasks])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        taskRefFile.to_csv(outFile)

    return taskRefFile


def build_participant_acc_data(subject, datadir, outdir):
    os.makedirs(outdir, exist_ok=True)

    accFile = os.path.join(outdir, f"{subject}.csv")
    if not os.path.exists(accFile):
        dataFiles = [
            pd.read_csv(
                build_patient_file_path(datadir, "GENEActiv", subject, i),
                delimiter="\t",
                index_col="timestamp",
                parse_dates=True,
                skipinitialspace=True,
                date_parser=parse_datetime_from_timestamp,
            )
            for i in range(1, 5)
        ]
        subjectFile = pd.concat(dataFiles).dropna().drop_duplicates()
        subjectFile = subjectFile / constants.g
        subjectFile.index.name = "time"
        subjectFile.rename(
            columns={
                "GENEActiv_X": "x",
                "GENEActiv_Y": "y",
                "GENEActiv_Z": "z",
                "GENEActiv_Magnitude": "mag",
            },
            inplace=True,
        )
        subjectFile.to_csv(accFile)

    else:
        print(f'Using saved subject accelerometery data at "{accFile}".')


def build_patient_file_path(dataFolder, device, subject_id, index):
    return os.path.join(
        dataFolder,
        device,
        get_patient_folder(subject_id),
        f"rawdata_day{index}.txt",
    )


def get_patient_folder(subject_id):
    subject_num, subject_loc = subject_id.split("_", 1)
    if subject_loc == "BOS":
        return "patient" + subject_num
    elif subject_loc == "NYC":
        return "patient" + subject_num + "_NY"
    else:
        raise AssertionError("Invalid subject id")


def label_acc_data(
    label, datadir=RAW_DIR, processeddir=PROCESSED_DIR, n_jobs=N_JOBS
):
    taskRefFile = build_task_reference_file(datadir, processeddir)
    subjects = taskRefFile["subject_id"].unique()

    outdir = os.path.join(processeddir, "raw_labels")
    os.makedirs(outdir, exist_ok=True)

    if len(glob(os.path.join(outdir, "*.csv"))) != len(subjects):
        taskDictionary = build_task_dictionary(datadir, processeddir)
        accdir = os.path.join(processeddir, "acc_data")

        Parallel(n_jobs=n_jobs)(
            delayed(label_participant_data)(
                subject, taskRefFile, taskDictionary, accdir, outdir, label
            )
            for subject in tqdm(subjects)
        )

    else:
        print("Label data already compiled...\n")


def build_task_dictionary(datadir=RAW_DIR, outdir=PROCESSED_DIR):
    processedDictionaryPath = os.path.join(outdir, "TaskDictionary.csv")

    if os.path.exists(processedDictionaryPath):
        taskDictionary = pd.read_csv(
            processedDictionaryPath, index_col="task_code"
        )
    else:
        os.makedirs(os.path.dirname(outdir), exist_ok=True)

        taskDictionary = pd.read_csv(
            os.path.join(datadir, "TaskCodeDictionary.csv")
        )
        taskDictionary["is-walking"] = taskDictionary["description"].apply(
            is_walking_given_description
        )
        taskDictionary["activity"] = taskDictionary["task_code"].apply(
            activity_given_task_code
        )
        taskDictionary.set_index("task_code", inplace=True)
        taskDictionary.to_csv(processedDictionaryPath)

    return taskDictionary


def is_walking_given_description(description):
    return (
        "walking"
        if (
            ("WALKING" in description.upper())
            or ("STAIRS" in description.upper())
        )
        else "not-walking"
    )


def activity_given_task_code(task_code):
    if "wlkg" in task_code:
        return "wlkg"

    elif "ftn" in task_code:
        return "ftn"

    elif "ram" in task_code:
        return "ram"

    else:
        return task_code


def label_participant_data(
    subject, taskRefFile, taskDictionary, accdir, outdir, label="is-walking"
):
    os.makedirs(outdir, exist_ok=True)
    labelFilePath = os.path.join(outdir, f"{subject}.csv")
    accFilePath = os.path.join(accdir, f"{subject}.csv")

    if not os.path.exists(labelFilePath):
        accFile = pd.read_csv(accFilePath, index_col=[0], parse_dates=True)

        participantTasks = taskRefFile[taskRefFile["subject_id"] == subject]

        accFile["annotation"] = -1
        accFile["day"] = -1

        for _, task in participantTasks.iterrows():
            startTime, endTime = task[["timestamp_start", "timestamp_end"]]
            mask = (accFile.index > startTime) & (accFile.index <= endTime)
            accFile.loc[mask, "annotation"] = taskDictionary.loc[
                task["task_code"], label
            ]
            accFile.loc[mask, "day"] = task["participant_day"]

        walkingLabels = accFile["annotation"]
        walkingLabels.to_csv(labelFilePath)

        accFile.to_csv(accFilePath)

    else:
        print(
            f'Using saved subject labelled accelerometery data at "{accFilePath}".'
        )


def download_ldopa(
    datadir, annot_label="is-walking", overwrite=False, n_jobs=10
):
    ldopa_datadir = os.path.join(datadir, "LDOPA_DATA")
    if overwrite or (
        len(glob(os.path.join(ldopa_datadir, "*.csv"))) < len(LDOPA_DOWNLOADS)
    ):
        print("Downloading data...\n")
        syn = synapseclient.login(USERNAME, apiKey=APIKEY)
        os.makedirs(ldopa_datadir, exist_ok=True)
        for tableName, tableId in tqdm(LDOPA_DOWNLOADS):
            syn.tableQuery(
                f"select * from {tableId}",
                includeRowIdAndRowVersion=False,
                downloadLocation=os.path.join(ldopa_datadir, tableName),
            )

        for tName, _ in tqdm(LDOPA_DOWNLOADS):
            copyfile(
                get_first_file(ldopa_datadir, tName),
                os.path.join(ldopa_datadir, f"{tName}.csv"),
            )
            rmtree(os.path.join(ldopa_datadir, tName))

    else:
        print(
            f'Using saved Levodopa Reponse study dictionary data at "{ldopa_datadir}".'
        )

    if len(glob(os.path.join(ldopa_datadir, "GENEActiv", "*"))) < 28:
        synapseutils.syncFromSynapse(syn, entity="syn20681023", path=datadir)

    else:
        print(
            f'Using saved Levodopa Reponse study accelerometery data at "{ldopa_datadir}".'
        )

    processeddir = os.path.join(datadir, "Ldopa_Processed")
    build_metadata(ldopa_datadir, processeddir)
    build_acc_data(ldopa_datadir, processeddir, n_jobs)
    label_acc_data(annot_label, ldopa_datadir, processeddir, n_jobs)


def load_data(
    datafile, sample_rate=100, index_col="time", annot_type="int"
):
    if ".parquet" in datafile:
        data = pd.read_parquet(datafile)
        data.dropna(inplace=True)

    else:
        data = pd.read_csv(
            datafile,
            index_col=index_col,
            parse_dates=[index_col],
            dtype={"x": "f4", "y": "f4", "z": "f4", "annotation": annot_type},
        )

    data, _ = actipy.process(data, sample_rate, verbose=False)

    return data


def resize(x, length, axis=1):
    length_orig = x.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    x = interp1d(t_orig, x, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )

    return x


def make_windows(
    data,
    winsec=10,
    sample_rate=100,
    resample_rate=30,
    label_type="threshold",
    dropna=True,
    verbose=False,
):
    X, Y, T, D = [], [], [], []

    for t, w in tqdm(
        data.resample(f"{winsec}s", origin="start"), disable=not verbose
    ):
        if len(w) < 1:
            continue

        t = t.to_numpy()

        x = w[["x", "y", "z"]].to_numpy()

        annot = w["annotation"]

        if pd.isna(annot).all():  # skip if annotation is NA
            continue

        if not is_good_window(x, sample_rate, winsec):  # skip if bad window
            continue

        if label_type == "mode":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Unable to sort modes"
                )
                mode_label = annot.mode(dropna=False).iloc[0]

                if mode_label == -1 or mode_label == "-1":
                    continue

                y = mode_label

                d = (
                    w["day"].mode(dropna=False).iloc[0]
                    if "day" in w.columns
                    else 1
                )

        if dropna and pd.isna(y):
            continue

        X.append(x)
        Y.append(y)
        T.append(t)
        D.append(d)

    X = np.stack(X)
    Y = np.stack(Y)
    T = np.stack(T)
    D = np.stack(D)

    if resample_rate != sample_rate:
        X = resize(X, int(resample_rate * winsec))

    return X, Y, T, D


def is_good_window(x, sample_rate, winsec):
    # Check window length is correct
    window_len = sample_rate * winsec
    if len(x) != window_len:
        return False

    # Check no nans
    if np.isnan(x).any():
        return False

    return True


def load_all_and_make_windows(datadir, outdir, n_jobs, overwrite=False):
    """Make windows from all available data, extract features and store locally"""
    if not overwrite and check_files_exist(
        outdir, ["X.npy", "Y.npy", "T.npy", "pid.npy", "day.npy"]
    ):
        print(f'Using files saved at "{outdir}".')
        return

    datafiles = glob(os.path.join(datadir, DATAFILES))

    Xs, Ys, Ts, Ds, Ps = zip(
        *Parallel(n_jobs=n_jobs)(
            delayed(load_and_make_windows)(datafile)
            for datafile in tqdm(datafiles, desc="Load all and make windows")
        )
    )

    X = np.vstack(Xs)
    Y = np.hstack(Ys)
    T = np.hstack(Ts)
    D = np.hstack(Ds)
    P = np.hstack(Ps)

    X, Y, T, D, P = filter_for_analysis(X, Y, T, D, P)

    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "X.npy"), X)
    np.save(os.path.join(outdir, "Y.npy"), Y)
    np.save(os.path.join(outdir, "day.npy"), D)
    np.save(os.path.join(outdir, "T.npy"), T)
    np.save(os.path.join(outdir, "pid.npy"), P)


def load_and_make_windows(datafile):
    X, Y, T, D = make_windows(
        load_data(datafile, **PROCESS_ARGS["load_data_args"]),
        **PROCESS_ARGS["make_windows_args"],
    )

    pid = Path(datafile)

    for _ in pid.suffixes:
        pid = Path(pid.stem)

    P = np.array([str(pid)] * len(X))

    return X, Y, T, D, P


def filter_for_analysis(X, Y, T, D, P):
    day_mask = (D == 1) | (D == 4)
    label_mask = (Y != "ram") & (Y != "ftn")

    X_out = X[day_mask & label_mask]
    Y_out = Y[day_mask & label_mask]
    T_out = T[day_mask & label_mask]
    D_out = D[day_mask & label_mask]
    P_out = P[day_mask & label_mask]

    return X_out, Y_out, T_out, D_out, P_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", "-d", default="data/ldopa/raw")
    parser.add_argument("--outdir", "-o", default="data/ldopa/prepared")
    parser.add_argument("--annot", "-a", default="activity")
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    download_ldopa(args.datadir, args.annot, args.overwrite, args.n_jobs)

    load_all_and_make_windows(
        args.datadir, args.outdir, args.n_jobs, args.overwrite
    )
