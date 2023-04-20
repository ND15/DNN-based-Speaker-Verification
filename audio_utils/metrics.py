import numpy as np
import matplotlib.pyplot as plt


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


def far(fn, tp):
    return fn / (fn + tp)


def frr(fp, tn):
    return fp / (tn + fp)


def calculate_eer(confusion_matrix):
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    print(FP, FN, TP, TN)
    # print(oe.inverse_transform(pred))

    fa = far(FN.sum(), TP.sum())

    fr = frr(FP.sum(), TN.sum())

    print("EER:", (fa + fr) / 2)

    return (fa + fr) / 2
