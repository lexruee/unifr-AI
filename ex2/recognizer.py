#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import wave
import numpy.fft as fft
import struct
import operator

__author__ = "Alexander Rüedlinger"
__license__ = """
/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * <a.rueedlinger@gmail.com> wrote this file. As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return Alexander Rüedlinger
 * ----------------------------------------------------------------------------
 */
"""

train_path = './sounds/train'
test_path = './sounds/finalTest'


def read_data_signal_from_wave_file(file_path):
    wav_file = wave.open(file_path, 'rb')
    n_frames = wav_file.getnframes()
    data_bytes = wav_file.readframes(n_frames)
    data = struct.unpack('%sb' % n_frames, data_bytes)
    wav_file.close()
    return np.array(data)


def read_data():
    train_files = ['%s/%s.wav' % (train_path, i) for i in np.arange(10)]
    test_files = ['%s/%s.wav' % (test_path, i) for i in np.arange(10)]
    train_data, test_data = [], []

    for train_file, test_file in zip(train_files, test_files):
        _train_data = read_data_signal_from_wave_file(train_file)
        _test_data = read_data_signal_from_wave_file(test_file)
        train_data.append(_train_data)
        test_data.append(_test_data)

    return train_data, test_data


def recognize(test_signal, train_data, user_key, **kwargs):
    stats = {}
    for key, train_signal in zip(range(0, 10), train_data):
        min_size = min(len(test_signal), len(train_signal))
        loss = 0
        # compute the loss function
        for i in range(0, min_size):
            distance = train_signal[i] - test_signal[i]
            loss += distance**2

        stats[key] = loss
    total_loss = 0
    for key, loss in stats.items():
        total_loss += loss

    stats = {k: (v / float(total_loss)) * 100 for k, v in stats.items()}

    min_key, min_loss = min(stats.iteritems(), key=operator.itemgetter(1))

    if kwargs.get('verbose', False):
        print('-------------------------------------------------------')
        print('Predict: %s' % min_key)
        print('User input: %s' % user_key)
        print('Min. loss: %s' % min_loss)
        print("success: %s" % (min_key == user_key))
        print('')
        for key, prop in stats.items():
            print("label: %s, loss: %s " % (key, round(prop, 2)))

        print('-------------------------------------------------------')

    if kwargs.get('plot', False):
        plt.figure()
        plt.bar(np.arange(10), stats.values(), align='center', alpha=0.5)
        plt.xticks(np.arange(10), stats.keys())
        plt.xlabel('Digits')
        plt.ylabel('Loss')
        for key, prop in stats.items():
            plt.text(key, prop, str(round(prop, 2)))

        plt.show()

    return min_key


def transform_data(data):
    return [abs(fft.rfft(signal * np.hamming(len(signal)))) for signal in data]


def main():
    train_data, test_data = read_data()
    train_data = transform_data(train_data)
    test_data = transform_data(test_data)
        # plt.figure(1)
        # plt.subplot(1,2,1)
        # plt.title(key)
        # plt.plot(tr_data_signal)
        # plt.subplot(1,2,2)
        # plt.title(key)
        # plt.plot(tt_data_signal)
        # plt.show()

    print("Minimalistic digit recognizer - Who needs a GUI?")

    loop, user_input = True, None
    while loop:
        user_input = input("Enter a digit from 0 to 9: ")
        user_input = int(user_input)
        loop = not (0 <= user_input <= 9)

    test_signal = test_data[user_input]
    recognize(test_signal, train_data, user_input, plot=True)


def compute_confusion_matrix():
    train_data, test_data = read_data()
    train_data = transform_data(train_data)
    test_data = transform_data(test_data)

    conf_mat = np.zeros((10, 10))  # actual / predicted

    for digit in range(0, 10):
        test_signal = test_data[digit]
        predict = recognize(test_signal, train_data, digit)
        conf_mat[digit, predict] += 1

    print("actual class / predicted")
    labels = " " * 4 + "  ".join(map(str, range(10)))
    print(labels)
    for i in range(10):
        s = str(i) + ' |'
        for j in range(10):
            s += " %s " % int(conf_mat[i, j])
        print(s)

if __name__ == '__main__':
    #main()
    compute_confusion_matrix()

