#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import apa
import os
from sklearn import svm

apa_data_path = "./apa"

train_nums = ['00', '01', '03', '04', '09', '10', '13', '14']
test_nums = ['15', '19', '20']
apa_train_dirs = ["%s/apa%s" % (apa_data_path, i) for i in train_nums]
apa_test_dirs = ["%s/apa%s" % (apa_data_path, i) for i in test_nums]


def get_apa_files(apa_dirs):
    files = []
    for apa_dir in apa_dirs:
        files += map(lambda s: os.path.join(apa_dir, s), os.listdir(apa_dir))

    return files

train_files = get_apa_files(apa_train_dirs)
test_files = get_apa_files(apa_test_dirs)

train_data, train_labels = apa.read_data_files(train_files)


def create_feature_vector(coordinates):
    X, Y = [], []
    for x, y in coordinates:
        X.append(x)
        Y.append(y)

    total_distance = 0
    for pair in zip(coordinates[0:len(coordinates)-2], coordinates[1::len(coordinates)-1]):
        a, b = pair
        distance = np.linalg.norm(a-b)
        total_distance += distance

    var_x = np.var(X)
    var_y = np.var(Y)
    mx = np.mean(X)
    my = np.mean(Y)
    n = len(coordinates)
    feature_vec = [var_x, var_y,  n, total_distance]
    return feature_vec


def print_conf_mat(conf_mat):
    print("actual class / predicted")
    labels = " " * 5 + (" " * 3).join(map(str, range(10)))
    print(labels)
    for i in range(10):
        s = str(i) + ' |'
        for j in range(10):
            s += " %2.0f " % int(conf_mat[i, j])
        print(s)


def main():
    classifier = svm.SVC()
    transformed_train_data = [create_feature_vector(coordinates) for coordinates in train_data]

    classifier.fit(transformed_train_data, train_labels)

    test_data, test_labels = apa.read_data_files(train_files)
    transformed_test_data = [create_feature_vector(coordinates) for coordinates in test_data]
    predicted = classifier.predict(transformed_test_data)

    conf_mat = np.zeros((10, 10))

    success, error = 0, 0
    for test_label, predicted_label in zip(test_labels, predicted):
        conf_mat[test_label, predicted_label] += 1
        if test_label == predicted_label:
            success += 1
        else:
            error += 1

    precision = success / float(success + error) * 100

    print("precision: %s " % precision)
    print_conf_mat(conf_mat)


if __name__ == '__main__':
    main()
