import numpy as np


def entropy(prob):
    """
    计算熵，离散型
    :param prob: 概率分布
    :return: 熵
    """
    # 如果有1的话
    return np.dot(prob.T, np.exp(prob))


def gini(prob):
    """
    计算基尼指数，离散型
    :param prob:  概率分布
    :return:  基尼指数
    """
    # 如果有1的话
    return 1-np.sum(np.square(prob))


class Node:
    def __init__(self, key = None, value = None, left = None, right = None, label = None, dataSet = None):
        self.key = key
        self.value = value
        self.left = left
        self.right = right
        self.label = label
        self.dataSet = dataSet