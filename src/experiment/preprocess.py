import pickle
import random

import numpy as np


def Z_ScoreNormalization(x):
    return (x - x.mean()) / x.std()


def norm(x):
    tmp_x = []
    for rri_rpa in x:
        split = np.hsplit(rri_rpa, 2)
        rri, rpa = split[0], split[1]
        rri = Z_ScoreNormalization(rri)
        rpa = Z_ScoreNormalization(rpa)
        tmp_x.append(np.hstack([rri, rpa]))
    return tmp_x


def load_zsfy_preprocessed_data():
    with open('../../output/preprocessed/fah-ecg.pkl', 'rb') as f:
        fah_ecg = pickle.load(f)
    x_train, x_train_5min, y_train = fah_ecg["o_train"], fah_ecg["o_train_5"], fah_ecg["y_train"]
    x_val, x_val_5min, y_val = fah_ecg["o_val"], fah_ecg["o_val_5"], fah_ecg["y_val"]
    x_test, x_test_5min, y_test, groups_test = fah_ecg["o_test"], fah_ecg["o_test_5"], fah_ecg["y_test"], \
                                               fah_ecg["groups_test"]

    # x_train = norm(x_train)
    # x_train_5min = norm(x_train_5min)
    # x_val = norm(x_val)
    # x_val_5min = norm(x_val_5min)
    # x_test = norm(x_test)
    # x_test_5min = norm(x_test_5min)

    # x_train.extend(x_val)
    # x_train_5min.extend(x_val_5min)
    # y_train.extend(y_val)

    x_train = np.array(x_train, dtype=np.float32)
    x_train_5min = np.array(x_train_5min, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    x_val = np.array(x_val, dtype=np.float32)
    x_val_5min = np.array(x_val_5min, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    x_test_5min = np.array(x_test_5min, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # seed = 7
    # random.seed(7)
    # idx_train = random.sample(range(len(x_train)), int(len(x_train) * 0.7))
    # num = [i for i in range(len(x_train))]
    # idx_val = set(num) - set(idx_train)
    # idx_val = list(idx_val)
    #
    # x_val = x_train[idx_val]
    # x_val_5min = x_train_5min[idx_val]
    # y_val = y_train[idx_val]
    #
    # x_train = x_train[idx_train]
    # x_train_5min = x_train_5min[idx_train]
    # y_train = y_train[idx_train]

    print('end: loading FAH-ECG preprocessed data\n')

    return x_train, x_train_5min, y_train, x_val, x_val_5min, y_val, x_test, x_test_5min, y_test, groups_test
