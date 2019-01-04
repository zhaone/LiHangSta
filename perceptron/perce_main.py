"""
2019/1/4
感知机算法主程序
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load(path):
    """
    :param path: 数据集路径
    :return:
    Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
    数据类型为np.float
    返回了所有的值，如果想取某类或者某个特征，自己取就OK了
    """
    tmp = np.loadtxt(path, dtype=np.str, delimiter=',')
    tmp = tmp[1:, :]
    # 三种花类别分别用 0 1 2表示
    tmp[np.argwhere(tmp[:, -1] == 'Iris-setosa'), -1] = '0'
    tmp[np.argwhere(tmp[:, -1] == 'Iris-versicolor'), -1] = '1'
    tmp[np.argwhere(tmp[:, -1] == 'Iris-virginica'), -1] = '2'
    return tmp.astype(np.float)


def preview(data):
    """
    查看数据集
    :param data: 数据源
    :return: none
    """
    # 选取epalLengthCm,SepalWidthCm特征以及类label
    data = data[:, [1, 2, 5]]
    f = plt.figure(1)
    plt.subplot(111)
    # 作图
    idx_1 = np.argwhere(data[:, -1] == 0)
    idx_2 = np.argwhere(data[:, -1] == 1)
    idx_3 = np.argwhere(data[:, -1] == 2)

    plt.scatter(data[idx_1, 0], data[idx_1, 1], marker='x', color='m', label='1', s=30)
    plt.scatter(data[idx_2, 0], data[idx_2, 1], marker='+', color='r', label='2', s=30)
    plt.scatter(data[idx_3, 0], data[idx_3, 1], marker='*', color='c', label='3', s=30)
    plt.show()
    # 可以看到类0和类2线性可分，感知机就用类0和1

def perce():
    """
    主程序
    :return: none
    """
    data = load('../data/perceptron/Iris.csv')
    # 准备数据
    # 用前两类的epalLengthCm,SepalWidthCm特征
    data = data[np.argwhere(data[:, -1] != 2), [1, 2, 5]]
    data[np.argwhere(data[:, -1] == 0), -1] = -1
    np.random.shuffle(data)
    X = data[:, :-1]
    Y = data[:, -1]
    # 首先画散点图
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    idx_1 = np.argwhere(data[:, -1] == -1)
    idx_2 = np.argwhere(data[:, -1] == 1)
    ax.scatter(data[idx_1, 0], data[idx_1, 1], marker='x', color='m', label='1', s=30)
    ax.scatter(data[idx_2, 0], data[idx_2, 1], marker='+', color='r', label='1', s=30)

    # 权重 偏执 学习率初始化
    w = np.random.rand(X.shape[1])*2-1
    b = np.random.rand()
    learning_rate = 0.1
    # 随机梯度下降开始
    terminate = False
    bingo = 0
    while not terminate:
        for x, y in zip(X, Y):
            # 梯度下降
            if (np.dot(w, x.T) + b)*y < 0:
                w = w+learning_rate*x*y
                b = b+learning_rate*y
                bingo = 0
            else:
                bingo = bingo + 1
                # 所有点都分类正确，退出循环
                if bingo == len(Y):
                    terminate = True
                    break
    # 画图检验结果
    # 计算两个点，(3,(-b - 3 * w[0]) / w[1])(8,(-b - 8 * w[0]) / w[1])
    p1 = np.array([3, 8])
    p2 = np.array([(-b - 3 * w[0]) / w[1], (-b - 8 * w[0]) / w[1]])
    # 画线
    ax.add_line(Line2D(p1, p2, linewidth=1, color='blue'))
    plt.plot()
    plt.show()
    print(w, b)


if __name__ == '__main__':
    perce()
