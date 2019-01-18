"""
2019/1/16
数据预处理函数，包括正规化和标准化
以下数据集每列为一个特征
这个目前还没用到
"""
import numpy as np


def normalize(data):
    """
    归一化 x = (x-min) / (max - min)
    :param data: 输入数据集
    :return:
    """
    # 先转置，返回副本
    norm_data = np.copy(data.T)
    # 求最大最小和距离
    _min, _max = np.min(data, axis=0), np.max(data, axis=0)
    diff = _max - _min
    # 防止除0错
    zero_index,  not_zero_index = diff == 0, diff != 0
    # 计算归一化后的值
    norm_data[zero_index, :] = 1
    norm_data[not_zero_index, :] = (norm_data[not_zero_index, :] - _min[not_zero_index])/diff[not_zero_index]

    return norm_data.T, _min, _max


def standardize(data):
    """
    对数据进行标准化， x = x-mean/standard deviation
    :param data: 数据集
    :return: std_data 标准化后的数据
    """
    # 返回副本
    std_data = np.zeros(data.shape)
    # 均值
    mean = data.mean(axis=0)
    # 标准差
    std = np.std(data, axis=0)
    for col in range(data.shape[1]):
        # 防止除0错
        if std[col] != 0:
            std_data[:, col] = (data[:, col] - mean)/std
    return std_data


def split_train_test(data, label, train_rate=0.8, shuffle=True):
    """
    将sample分为训练集和测试集
    :param data:  数据集特征向量
    :param label:  标签
    :param train_rate:  训练数据所占的比重
    :param shuffle:  是否shuffle
    :return: 训练集数据，测试机数据，训练集标签，测试集标签
    """
    _data = np.copy(data)
    train_num = int(_data.shape[1]*train_rate)
    train_data, train_label = _data[:, :train_num], label[:train_num]
    test_data, test_label = _data[:, train_num:], label[train_num:]
    return train_data, test_data, train_label, test_label
