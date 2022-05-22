# coding=utf-8
import pandas as pd
import numpy as np
from plt import pltShow
from kmeans import KMeans
import matplotlib.pyplot as plt

maxIterCnt = 50

def getData():
    filePath = "D:\dataset\hcvdat0.csv"
    data = pd.read_csv(filePath)
    maleId = data['Sex'] == 'm'
    femaleId = data['Sex'] == 'f'
    data['Sex'][maleId] = 1
    data['Sex'][femaleId] = 2
    data = data.dropna(axis=0, how = 'any')
    types = ['0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis']
    return data, types


def initMethodTest():
    data, types = getData()
    K = 5
    numOfData = data.shape[0]
    x_train = data.iloc[:, 2:14].values.reshape(numOfData, 12)
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
    # x轴代表ALB
    # y轴代表ALP
    pltShow(data, 4, 5, 1, closerstCenter1, center1, types)
    pltShow(data, 4, 5, 1, closerstCenter2, center2, types)
    pltShow(data, 4, 5, 1, closerstCenter3, center3, types)


def kTest():
    data, iris_types = getData()
    len = 20
    J1 = np.zeros((len, 1))
    J2 = np.zeros((len, 1))
    J3 = np.zeros((len, 1))
    x_train = data.iloc[:, 2:14].values.reshape(data.shape[0], 12)

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
    # initMethodTest()
    kTest()
