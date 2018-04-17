import numpy as np
from math import log
import sys
import matplotlib.pyplot as plt
from sklearn import svm

__author__ = 'Chongye Wang'

def read_data(file):
    data = []
    lines = file.readlines()
    i = 1
    matrix = []
    offset = 0
    while i < len(lines):
        lines[i] = lines[i].rstrip()
        char_list = [x for x in lines[i]]
        matrix.append(char_list)
        i += 1
        offset += 1
        if offset == 68:
            i += 2
            offset = 0
            data.append(matrix)
            matrix = []
    return data


def read_label(file):
    label = []
    lines = file.readlines()
    for line in lines:
        label.append(int(line[0]))
    return label


def trasfer_to_matrix(data):
    for idx in range(0, len(data)):
        face = []
        for i in range(0, 68):
            face.append([])
            for j in range(0, 59):
                face[i].append(0)
        for i in range(0, len(data[idx])):
            for j in range(0, len(data[idx][i])):
                if data[idx][i][j] == '#':
                    face[i][j] = 1
        list = []
        for line in face:
            list.extend(line)
        data[idx] = list


def get_accuracy(estimate, real):
    count = 0.0
    for i in range(0, len(real)):
        if estimate[i] == real[i]: count += 1
    return count / len(real)


if __name__ == "__main__":

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    with open("facedatatrain.txt") as file:
        train_data = read_data(file)
    with open("facedatatest.txt") as file:
        test_data = read_data(file)
    with open("facedatatrainlabels.txt") as file:
        train_label = read_label(file)
    with open("facedatatestlabels.txt") as file:
        test_label = read_label(file)

    trasfer_to_matrix(train_data)
    trasfer_to_matrix(test_data)

    clf = svm.SVC(gamma=0.002, C=100)
    clf.fit(train_data, train_label)
    test_result = clf.predict(test_data)
    print(get_accuracy(test_result, test_label))
    #0.926666666667
