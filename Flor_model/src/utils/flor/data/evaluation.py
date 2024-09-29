# ./flor/data/evaluation.py

"""
Tool to metrics calculation through data and label (string and string).
 * Calculation from Optical Character Recognition (OCR) metrics with editdistance.
"""

import string
import unicodedata
import editdistance
import numpy as np


def ocr_metrics(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    """Calculate Character Error Rate (CER), Word Error Rate (WER) and Sequence Error Rate (SER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return (1, 1, 1)

    cer, wer, ser = [], [], []

    for (pd, gt) in zip(predicts, ground_truth):
        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("ASCII", "ignore").decode("ASCII")
            gt = unicodedata.normalize("NFKD", gt).encode("ASCII", "ignore").decode("ASCII")

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        pd_cer, gt_cer = list(pd), list(gt)
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (max(len(pd_cer), len(gt_cer))))

        pd_wer, gt_wer = pd.split(), gt.split()
        dist = editdistance.eval(pd_wer, gt_wer)
        wer.append(dist / (max(len(pd_wer), len(gt_wer))))

        pd_ser, gt_ser = [pd], [gt]
        dist = editdistance.eval(pd_ser, gt_ser)
        ser.append(dist / (max(len(pd_ser), len(gt_ser))))

    metrics = [cer, wer, ser]
    metrics = np.mean(metrics, axis=1)

    return metrics


def cer_only(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    """Calculate only Character Error Rate (CER)"""

    if len(predicts) == 0 or len(ground_truth) == 0:
        return 1

    cer = []

    for (pd, gt) in zip(predicts, ground_truth):
        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("ASCII", "ignore").decode("ASCII")
            gt = unicodedata.normalize("NFKD", gt).encode("ASCII", "ignore").decode("ASCII")

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        pd_cer, gt_cer = list(pd), list(gt)
        dist = editdistance.eval(pd_cer, gt_cer)
        cer.append(dist / (len(gt_cer)))

    cer_mean = np.mean(cer)
    cer_mean = round(cer_mean * 100, 3)

    return cer_mean


def wer_only(predicts, ground_truth, norm_accentuation=False, norm_punctuation=False):
    """
    Calculate only Word Error Rate (WER) between predictions and ground truth.

    Args:
        predicts (list of str): List of predicted sentences.
        ground_truth (list of str): List of ground truth sentences.
        norm_accentuation (bool): Normalize to ignore accentuation differences if True.
        norm_punctuation (bool): Remove punctuation before computing WER if True.

    Returns:
        float: The mean WER across all predictions.
    """

    if len(predicts) == 0 or len(ground_truth) == 0:
        return 100.0  # Return maximum WER percentage if inputs are empty

    wer = []

    for pd, gt in zip(predicts, ground_truth):
        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("ASCII", "ignore").decode("ASCII")
            gt = unicodedata.normalize("NFKD", gt).encode("ASCII", "ignore").decode("ASCII")

        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        # Split the sentences into words for WER calculation
        pd_words, gt_words = pd.split(), gt.split()

        # Compute the Levenshtein distance between the predicted and ground truth word lists
        dist = editdistance.eval(pd_words, gt_words)

        # Calculate WER as the ratio of the distance to the length of the longer sentence
        wer.append(dist / (len(gt_words)))

    # Calculate the mean WER and multiply by 100 to express it as a percentage
    mean_wer = np.mean(wer) * 100

    # Round the mean WER to 3 decimal places
    return round(mean_wer, 3)
