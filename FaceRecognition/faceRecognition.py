import numpy as np
from math import log
import sys
import matplotlib.pyplot as plt

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
                face[i].append(' ')
        for i in range(0, len(data[idx])):
            for j in range(0, len(data[idx][i])):
                face[i][j] = data[idx][i][j]
        data[idx] = face


def generate_classifier(data, label):
    classifier = {}
    classifier[0] = []
    classifier[1] = []
    for i in range(0, len(label)):
        classifier[label[i]].append(data[i])
    return classifier


def training(classifier):
    result = {}
    for index in range(0, 2):
        result[index] = {}
        for i in range(0, 68):
            for j in range(0, 59):
                count = 0.0
                for face in classifier[index]:
                    if face[i][j] == '#':
                        count += 1
                prob = (count + 0.1) / (len(classifier[index]) + 0.1 * 2)
                result[index][(j, i)] = prob
    return result


def testing(data, training_result):
    result = {}
    result[0] = []
    result[1] = []
    for face in data:
        face_value = {}
        for index in range(0, 2):
            map = log(0.5, 2)
            for i in range(0, 68):
                for j in range(0, 59):
                    if face[i][j] == '#':
                        map += log(training_result[index][(j, i)], 2)
                    else:
                        map += log(1 - training_result[index][(j, i)], 2)
            face_value[index] = map
        if face_value[0] > face_value[1]:
            result[0].append(face)
        else:
            result[1].append(face)
    return result


def transfer_digit(classifier):
    result = {}
    for index in classifier:
        for face in classifier[index]:
            list = []
            for line in face:
                list.extend(line)
            result[tuple(list)] = index
    return result


def get_accuracy(real_digit, estimate_digit, target_digit, size):
    count = 0.0
    for face in real_digit:
        if real_digit[face] == target_digit:
            estimate = estimate_digit[face]
            if target_digit == estimate:
                count += 1
    return count / size


def generate_oddsRatio_matrix(class1, class2, training_result):
    matrix1 = np.zeros((68, 59))
    matrix2 = np.zeros((68, 59))
    odds_ratio = np.zeros((68, 59))
    for ele in training_result[class1]:
        x = ele[0]
        y = ele[1]
        matrix1[y][x] = log(training_result[class1][ele], 2)
    for ele in training_result[class2]:
        x = ele[0]
        y = ele[1]
        matrix2[y][x] = log(training_result[class2][ele], 2)
    for i in range(68):
        for j in range(59):
            v1 = training_result[class1][(j, i)]
            v2 = training_result[class2][(j, i)]
            ratio = log(v1 / v2, 2)
            odds_ratio[i][j] = ratio
    return matrix1, matrix2, odds_ratio



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

    #num of row 68  num of column 59
    trasfer_to_matrix(train_data)
    trasfer_to_matrix(test_data)

    #transfer to dictionary classfier
    train_classifier = generate_classifier(train_data, train_label)
    test_classifier = generate_classifier(test_data, test_label)

    #get trainging result and estimate result
    training_result = training(train_classifier)
    estimated_result = testing(test_data, training_result)

    #convert to face-digit dictionary
    testing_classifier_digit = transfer_digit(test_classifier)
    estimated_digit = transfer_digit(estimated_result)

    #get the accuracy
    face_accuracy = get_accuracy(testing_classifier_digit, estimated_digit, 1, len(test_classifier[1]))
    non_face_accuracy = get_accuracy(testing_classifier_digit, estimated_digit, 0, len(test_classifier[0]))

    #print result
    print("Face accuracy: " + str(face_accuracy)) #0.86301369863
    print("Non face accuracy: " + str(non_face_accuracy)) #0.896103896104

    #plot
    fig = plt.figure()
    class1 = 1
    class2 = 0
    matrix1, matrix2, odds_ratio = generate_oddsRatio_matrix(class1, class2, training_result)
    ax = fig.add_subplot(1,3,1)
    cax = ax.matshow(matrix1, interpolation = 'nearest')
    fig.colorbar(cax)
    ax = fig.add_subplot(1,3,2)
    cax = ax.matshow(matrix2, interpolation = 'nearest')
    fig.colorbar(cax)
    ax = fig.add_subplot(1,3,3)
    cax = ax.matshow(odds_ratio, interpolation = 'nearest')
    fig.colorbar(cax)
    plt.show()
