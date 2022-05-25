import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F


def summarise_scores(scores, name=None, average=False):
    """Compute median and interquartile range of scores
    and put it in a nice format, e.g. 0.83 (0.80, 0.85)"""
    scores = pd.DataFrame(scores)
    if average:  # average across labels
        scores = scores.mean(1).to_frame()
    scores = scores.quantile((0.5, 0.25, 0.75))
    scores = scores.round(2).applymap("{:.2f}".format).T
    scores = scores[0.5] + " (" + scores[0.25] + ", " + scores[0.75] + ")"
    scores.name = name
    return scores


def summarise_epoch_scores(scores):
    """Compute mean and +- std and put them in a row"""
    scores = pd.DataFrame(scores)
    avg_epoch_class_score = scores.mean(1).to_numpy()

    return avg_epoch_class_score


def classification_scores(Y_test, Y_test_pred):
    cohen_kappa = metrics.cohen_kappa_score(Y_test, Y_test_pred)
    precision = metrics.precision_score(
        Y_test, Y_test_pred, average="macro", zero_division=0
    )
    recall = metrics.recall_score(
        Y_test, Y_test_pred, average="macro", zero_division=0
    )
    f1 = metrics.f1_score(
        Y_test, Y_test_pred, average="macro", zero_division=0
    )

    return cohen_kappa, precision, recall, f1


def save_report(
    precision_list, recall_list, f1_list, cohen_kappa_list, report_path
):
    data = {
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "kappa": cohen_kappa_list,
    }

    df = pd.DataFrame(data)
    df.to_csv(report_path, index=False)


def classification_report(results, report_path):
    # logger is a tf logger
    # Collate metrics
    cohen_kappa_list = [result[0] for result in results]
    precision_list = [result[1] for result in results]
    recall_list = [result[2] for result in results]
    f1_list = [result[3] for result in results]

    save_report(
        precision_list, recall_list, f1_list, cohen_kappa_list, report_path
    )


def regression_scores(Y_test, Y_test_pred):
    r2 = metrics.r2_score(Y_test, Y_test_pred)
    rmse = metrics.mean_squared_error(
        Y_test, Y_test_pred, squared=False
    )  # if False, returns RMSE
    return {"r2": r2, "rmse": rmse}


def regression_report(results, logger):
    # Collate metrics
    r2_list = [result["r2"] for result in results]
    rmse_list = [result["rmse"] for result in results]

    logger.info(
        "\nRegression scores:"
        "\n R^2: "
        + str(summarise_scores(r2_list, average=True).item())
        + "\nRMSE: "
        + str(summarise_scores(rmse_list, average=True).item())
    )


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(
                bin_upper.item()
            )
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += (
                    torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                    * prop_in_bin
                )

        return ece
