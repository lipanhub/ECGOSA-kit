import os

import wfdb

# https://physionet.org/physiobank/database/apnea-ecg/
data_dir = '../../resources/apnea-ecg-database-1.0.0'
# number of threads to preprocess recording
num_worker = 35


def sta_mix():
    print('start: preprocess released set')

    released_set_recording_names = [
        'a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10',
        'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20',
        'b01', 'b02', 'b03', 'b04', 'b05',
        'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10'
    ]

    label_list = []

    for i in range(len(released_set_recording_names)):
        recording_name = released_set_recording_names[i]
        labels = wfdb.rdann(os.path.join(data_dir, recording_name), extension='apn').symbol
        label_list.append(labels)

    print('start: preprocess withheld set')

    withheld_set_recording_names = [
        'x01', 'x02', 'x03', 'x04', 'x05', 'x06', 'x07', 'x08', 'x09', 'x10',
        'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
        'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30',
        'x31', 'x32', 'x33', 'x34', 'x35'
    ]
    answers = {}
    with open(os.path.join(data_dir, 'event-2-answers'), 'r') as f:
        for answer in f.read().split('\n\n'):
            answers[answer[:3]] = list(''.join(answer.split()[2::2]))

    for i in range(len(withheld_set_recording_names)):
        recording_name = withheld_set_recording_names[i]
        labels = answers[recording_name]
        label_list.append(labels)

    num_same = 0
    num_mix = 0

    for i in range(len(label_list)):
        labels = label_list[i]
        for j in range(len(labels)):
            if j < 2 or j >= len(labels) - 2:
                continue
            if labels[j - 2] == 'N' and labels[j - 1] == 'N' and labels[j] == 'N' and labels[j + 1] == 'N' and labels[
                j + 2] == 'N':
                num_same = num_same + 1
            elif labels[j - 2] == 'A' and labels[j - 1] == 'A' and labels[j] == 'A' and labels[j + 1] == 'A' and labels[
                j + 2] == 'A':
                num_same = num_same + 1
            else:
                num_mix = num_mix + 1

    print(num_same / (num_same + num_mix))
    print(num_mix / (num_same + num_mix))
    print()


if __name__ == '__main__':
    sta_mix()
