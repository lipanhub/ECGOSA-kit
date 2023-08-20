import csv

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import keras

import src.infra.metric as metrics
from src.experiment.util import plot_and_save_cfm


# LeNet5论文计算方式的结果（计算per segment不管丢弃段，计算true AHI关心丢弃段但pred AHI不用关心）
def final_test_LeNet5(log_dir, x_test, x_test_5min, y_test, recording_name_test):
    # load trained model
    weights_filepath = log_dir + '/checkpoint/classifier.weights.best.hdf5'
    custom_objects = {'tf': tf}
    model = keras.models.load_model(weights_filepath, custom_objects=custom_objects)

    # save prediction score
    y_score = model.predict([x_test, x_test_5min], batch_size=32, verbose=1)
    y_pred = np.argmax(y_score, axis=-1)
    # y_pred = model.predict([x_test, x_test_5min], batch_size=1024, verbose=1)
    # y_pred = np.int64(y_pred >= 0.5).flatten()

    # per segment performance
    print('参考LeNet-5的结果')
    cfm, acc, sn, sp, f1 = metrics.per_segment(y_test, y_pred)
    print("acc: {}, sn: {}, sp: {}, f1: {}".format(acc * 100, sn * 100, sp * 100, f1))
    label_names = ['SA', 'NSA']
    # cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
    plot_and_save_cfm(log_dir, cfm, 'cfm-per-segment', label_names)
    # per recording performance
    with open("../../resources/apnea-ecg-database-1.0.0/additional-information.txt", "r") as f:
        original = []
        for line in f:
            rows = line.strip().split("\t")
            if len(rows) == 12:
                if rows[0].startswith("x"):
                    original.append([rows[0], float(rows[3]) / float(rows[1]) * 60])

    original_AHI = pd.DataFrame(original, columns=["subject", "true_AHI"])
    original_AHI = original_AHI.set_index("subject")
    original_AHI.name = 'true_AHI'

    per_segment_pred = pd.DataFrame({"y_pred": y_pred, "subject": recording_name_test})
    predict_AHI = per_segment_pred.groupby(by="subject").apply(lambda d: d["y_pred"].mean() * 60)
    predict_AHI.name = 'predict_AHI'

    true_pred_AHI = pd.concat([original_AHI, predict_AHI], axis=1)

    corr = true_pred_AHI.corr()

    true_pred_AHI = true_pred_AHI.applymap(lambda a: int(a > 5))

    cfm = confusion_matrix(true_pred_AHI['true_AHI'], true_pred_AHI['predict_AHI'], labels=(1, 0))

    plot_and_save_cfm(log_dir, cfm, 'cfm-per-recording', label_names)

    TP, TN, FP, FN = cfm[0, 0], cfm[1, 1], cfm[1, 0], cfm[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    print("acc: {}, sn: {}, sp: {}, corr: {}".format(acc * 100, sn * 100, sp * 100, corr['true_AHI']['predict_AHI']))


# 不关心丢弃段的结果（per segment、计算true AHI和pred AHI均不关心丢弃段）
def final_test_without_discarded_segment(log_dir, x_test, x_test_5min, y_test, recording_name_test):
    # load trained model
    weights_filepath = log_dir + '/checkpoint/classifier.weights.best.hdf5'
    custom_objects = {'tf': tf}
    model = keras.models.load_model(weights_filepath, custom_objects=custom_objects)

    # save prediction score
    y_score = model.predict([x_test, x_test_5min], batch_size=32, verbose=1)
    y_pred = np.argmax(y_score, axis=-1)
    # y_pred = model.predict([x_test, x_test_5min], batch_size=1024, verbose=1)
    # y_pred = np.int64(y_pred >= 0.5).flatten()

    # per segment performance
    print('计算原始AHI时忽略丢弃段的结果')
    cfm, acc, sn, sp, f1 = metrics.per_segment(y_test, y_pred)
    print("acc: {}, sn: {}, sp: {}, f1: {}".format(acc * 100, sn * 100, sp * 100, f1))
    label_names = ['Non Apnea', 'Apnea']
    # cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
    plot_and_save_cfm(log_dir, cfm, 'cfm-per-segment-2', label_names)
    # per recording performance
    per_segment_result = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "subject": recording_name_test})

    with open('/GraduateStudents/cs601/lipan/project/SeCon-zsfy/resources/AHI.csv', 'r') as AHI_file:
        original = list(csv.reader(AHI_file))
    for item in original:
        item[1] = float(item[1])
    original = pd.DataFrame(original, columns=["subject", "true_AHI"])
    true_AHI = original.set_index("subject")
    true_AHI.name = 'true_AHI'

    predict_AHI = per_segment_result.groupby(by="subject").apply(lambda d: d["y_pred"].mean() * 60)
    predict_AHI.name = 'predict_AHI'

    true_pred_AHI = pd.concat([true_AHI, predict_AHI], axis=1)

    corr = true_pred_AHI.corr()

    true_pred_AHI = true_pred_AHI.applymap(lambda a: int(a >= 5))

    cfm = confusion_matrix(true_pred_AHI['true_AHI'], true_pred_AHI['predict_AHI'], labels=(1, 0))

    plot_and_save_cfm(log_dir, cfm, 'cfm-per-recording-2', label_names)

    TP, TN, FP, FN = cfm[0, 0], cfm[1, 1], cfm[1, 0], cfm[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    print("acc: {}, sn: {}, sp: {}, corr: {}".format(acc * 100, sn * 100, sp * 100, corr['true_AHI']['predict_AHI']))


def predict_discarded_segment(pred_model, discarded_flag):
    result_y_true = []
    idx = 0
    for i in range(len(discarded_flag)):
        if discarded_flag[i]:
            result_y_true[i] = 0
        else:
            result_y_true[i] = pred_model[idx]
            idx = idx + 1
    return result_y_true


# 丢弃段预测为1的结果
def final_test_predict_discarded_segment(log_dir, x_test, x_test_5min, y_test_with_discarded_segment,
                                         recording_name_test_with_discarded_segment,
                                         discarded_flag):
    # load trained model
    weights_filepath = log_dir + '/checkpoint/classifier.weights.best.hdf5'
    custom_objects = {'tf': tf}
    model = keras.models.load_model(weights_filepath, custom_objects=custom_objects)

    # save prediction score
    y_score = model.predict([x_test, x_test_5min], batch_size=1024, verbose=1)
    pred_model = np.argmax(y_score, axis=-1)

    y_pred = predict_discarded_segment(pred_model, discarded_flag)

    print('丢弃段预测为1的结果')
    cfm, acc, sn, sp, f1 = metrics.per_segment(y_test_with_discarded_segment, y_pred)
    print("acc: {}, sn: {}, sp: {}, f1: {}".format(acc * 100, sn * 100, sp * 100, f1))
    label_names = ['SA', 'NSA']
    # cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
    plot_and_save_cfm(log_dir, cfm, 'cfm-per-recording-2', label_names)

    # per recording performance
    per_segment_result = pd.DataFrame({"y_true": y_test_with_discarded_segment, "y_pred": y_pred,
                                       "subject": recording_name_test_with_discarded_segment})

    true_AHI = per_segment_result.groupby(by="subject").apply(lambda d: d["y_true"].mean() * 60)
    true_AHI.name = 'true_AHI'
    predict_AHI = per_segment_result.groupby(by="subject").apply(lambda d: d["y_pred"].mean() * 60)
    predict_AHI.name = 'predict_AHI'

    true_pred_AHI = pd.concat([true_AHI, predict_AHI], axis=1)

    corr = true_pred_AHI.corr()

    true_pred_AHI = true_pred_AHI.applymap(lambda a: int(a > 5))

    cfm = confusion_matrix(true_pred_AHI['true_AHI'], true_pred_AHI['predict_AHI'], labels=(1, 0))
    TP, TN, FP, FN = cfm[0, 0], cfm[1, 1], cfm[1, 0], cfm[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    print("acc: {}, sn: {}, sp: {}, corr: {}".format(acc * 100, sn * 100, sp * 100, corr['true_AHI']['predict_AHI']))
