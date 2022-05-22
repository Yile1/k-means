import matplotlib.pyplot as plt

def pltShow(data, x, y, labelIndex, closerstCenter, center, types):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for type in types:
        dataCopy = data.iloc[(data.iloc[:, labelIndex] == type).tolist()]
        x_axis = dataCopy.iloc[:, x]
        y_axis = dataCopy.iloc[:, y]
        plt.scatter(x_axis, y_axis, label = type)
    plt.title('label known')
    plt.legend()
    plt.subplot(1, 2, 2)
    for index, centerPoint in enumerate(center):
        currentIndex = (closerstCenter == index).flatten()
        dataCopy = data.iloc[currentIndex]
        plt.scatter(dataCopy.iloc[:, x], dataCopy.iloc[:, y], label=index)
    for index, centerPoint in enumerate(center):
        plt.scatter(centerPoint[x], centerPoint[y], c = "black", marker='x')
    plt.legend()
    plt.title('label kmeans')
    plt.show()