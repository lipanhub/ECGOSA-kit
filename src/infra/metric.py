import csv
import os

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score


def per_segment(y_true, y_pred):
    # 这里要改回labels=(0, 1)，然后TP为[1,1]
    cfm = confusion_matrix(y_true, y_pred, labels=(1, 0))
    TP, TN, FP, FN = cfm[0, 0], cfm[1, 1], cfm[1, 0], cfm[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    f1 = f1_score(y_true, y_pred, average='binary')
    return cfm, acc, sn, sp, f1


def per_recording():
    base_dir = "output"

    # Table 2
    output = []
    methods = ["SE-MSCNN"]
    for method in methods:
        df = pd.read_csv(os.path.join(base_dir, "%s.csv" % method), header=0)
        df["y_pred"] = df["y_score"] > 0.5
        df = df.groupby(by="subject").apply(lambda d: d["y_pred"].mean() * 60)
        df.name = method
        output.append(df)
    output = pd.concat(output, axis=1)
    with open(Constant.ZSFY_TEST_AHI_FILE_PATH, 'r') as AHI_file:
        original = list(csv.reader(AHI_file))
    for item in original:
        item[1] = float(item[1])
    original = pd.DataFrame(original, columns=["subject", "original"])
    original = original.set_index("subject")
    all = pd.concat((output, original), axis=1)
    corr = all.corr()
    all1 = all.applymap(lambda a: int(a > 5))
    result = []
    for method in methods:
        C = confusion_matrix(all1["original"], all1[method], labels=(1, 0))
        TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
        acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
        auc = roc_auc_score(all["original"] > 5, all[method])
        result.append([method, acc * 100, sn * 100, sp * 100, auc, corr["original"][method]])
        print("acc: {}, sn: {}, sp: {}, auc: {},corr: {}".format(acc * 100, sn * 100, sp * 100, auc,
                                                                 corr["original"][method]))
