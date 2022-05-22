from random import random

import numpy as np
from sklearn import metrics
# KMeans
class KMeans:
    def __init__(self, data, K):
        self.data = data
        self.K = K

    def train(self, maxIterCnt, initMethod):
        if initMethod == 1:
            center = KMeans.centerInit1(self.data, self.K)
        elif initMethod == 2:
            center = KMeans.centerInit2(self.data, self.K)
        else:
            center = KMeans.centerInit3(self.data, self.K)

        numOfData = self.data.shape[0]
        closestCenter = np.empty((numOfData, 1))

        for i in range(maxIterCnt):
            closestCenter = KMeans.findNeighborCenter(self.data, center)
            center = KMeans.centerUpdate(self.data, closestCenter, self.K)
        return center, closestCenter

    def centerInit1(data, K):
        numOfData = data.shape[0]
        randomId = np.random.permutation(numOfData)

        center = data[randomId[:K], :]
        return center

    def centerInit2(data, K):
        numOfData, featureNum = np.shape(data)
        center = np.zeros((K, featureNum))

        index = np.random.randint(0, numOfData)
        center[0] = data[index]

        for i in range(1, K):
            closestCenter = KMeans.findNeighborCenter(data, center)
            distance = np.zeros((numOfData, 1))
            for j in range(numOfData):
                centerId = int(closestCenter[j][0])
                distance[j] = np.sum((data[j, :] - center[centerId, :]) ** 2)

            centerId = np.argmax(distance)
            center[i] = data[centerId]
        return center

    def centerInit3(data, K):
        numOfData, featureNum = np.shape(data)
        center = np.zeros((K, featureNum))

        # 随机选择一个样本点为第一个聚类中心
        index = np.random.randint(0, numOfData)
        center[0] = data[index]

        for i in range(1, K):
            closestCenter = KMeans.findNeighborCenter(data, center)
            distance = np.zeros((numOfData, 1))
            totalSum = 0
            for j in range(numOfData):
                centerId = int(closestCenter[j][0])
                distance[j] = np.sum((data[j, :] - center[centerId, :]) ** 2)
                totalSum += distance[j]

            totalSum *= np.random.random()
            # 以概率获得距离最远的样本点作为聚类中心
            for id, dist in enumerate(distance):
                totalSum -= dist
                if totalSum > 0:
                    continue
                center[i] = data[id]
                break
        return center

    def findNeighborCenter(data, center):
        numOfData = data.shape[0]
        K = center.shape[0]
        closestCenter = np.zeros((numOfData, 1))

        for i in range(numOfData):
            distance = np.zeros((K, 1))
            for k in range(K):
                diff = data[i, :] - center[k, :]
                distance[k] = np.sum(diff ** 2)
            closestCenter[i] = np.argmin(distance)

        return closestCenter

    def centerUpdate(data, closestCenter, K):
        featureNum = data.shape[1]
        center = np.zeros((K, featureNum))
        for k in range(K):
            clusterId = closestCenter == k
            center[k] = np.mean(data[clusterId.flatten(), :], axis=0)

        return center

    def computeNMI(numOfData, closerstCenter, K):
        label = np.zeros((1, numOfData))
        for i in range(K):
            start = int(numOfData * i / K)
            end = int(numOfData * (i + 1) / K)
            for j in range(start, end):
                label[0, j] = i
        return metrics.normalized_mutual_info_score(label.flatten(), closerstCenter.T.flatten())

    def computeJ(self, maxIterCnt, initMethod):
        center, closestCenter = self.train(maxIterCnt, initMethod)
        totalDiff = 0
        for i in range(closestCenter.shape[0]):
            centerId = int(closestCenter[i][0])
            totalDiff += np.sum((self.data[i, :] - center[centerId, :]) ** 2)

        return totalDiff