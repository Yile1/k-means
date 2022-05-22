# coding=utf-8
import pandas as pd
import numpy as np
from plt import pltShow
from kmeans import KMeans
import matplotlib.pyplot as plt

maxIterCnt = 50


def getData():
    filePath = "D:\dataset\heart_failure_clinical_records_dataset.csv"
    data = pd.read_csv(filePath)
    iris_types = [0, 1]
    return data, iris_types


def initMethodTest():
    data, iris_types = getData()
    K = 2
    numOfData = data.shape[0]
    x_train = data.iloc[:, 0:12].values.reshape(numOfData, 12)
    k_means = KMeans(x_train, K)
    center1, closerstCenter1 = k_means.train(maxIterCnt, 1)
    center2, closerstCenter2 = k_means.train(maxIterCnt, 2)
    center3, closerstCenter3 = k_means.train(maxIterCnt, 3)
    nmi1 = KMeans.computeNMI(numOfData, closerstCenter1, K)
    print("nmi1 is %f" % nmi1)
    nmi2 = KMeans.computeNMI(numOfData, closerstCenter2, K)
    print("nmi2 is %f" % nmi2)
    nmi3 = KMeans.computeNMI(numOfData, closerstCenter3, K)
    print("nmi3 is %f" % nmi3)
    pltShow(data, 2, 4, 12, closerstCenter1, center1, iris_types)
    pltShow(data, 2, 4, 12, closerstCenter2, center2, iris_types)
    pltShow(data, 2, 4, 12, closerstCenter3, center3, iris_types)


def kTest():
    data, iris_types = getData()
    len = 20
    J1 = np.zeros((len, 1))
    J2 = np.zeros((len, 1))
    J3 = np.zeros((len, 1))
    x_train = data.iloc[:, 0:12].values.reshape(data.shape[0], 12)

    for k in range(2, 22):
        k_means = KMeans(x_train, k)
        J1[k - 2][0] = k_means.computeJ(maxIterCnt, 1)
        J2[k - 2][0] = k_means.computeJ(maxIterCnt, 2)
        J3[k - 2][0] = k_means.computeJ(maxIterCnt, 3)

    x_axis = [i for i in range(2, 22)]

    plt.plot(x_axis, J1, color='red', label='J1')
    plt.plot(x_axis, J2, color='green', label='J2')
    plt.plot(x_axis, J3, color='blue', label='J3')
    plt.legend()  # 显示图例
    plt.xlabel('iteration k')
    plt.ylabel('J')
    plt.show()


if __name__ == '__main__':
    initMethodTest()
    # kTest()
