import numpy as np
from knn.KNearestNeighbor import Knn


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
    tmp = tmp[:, 1:]
    # 三种花类别分别用 0 1 2表示
    tmp[np.argwhere(tmp[:, -1] == 'Iris-setosa'), -1] = '0'
    tmp[np.argwhere(tmp[:, -1] == 'Iris-versicolor'), -1] = '1'
    tmp[np.argwhere(tmp[:, -1] == 'Iris-virginica'), -1] = '2'
    return tmp.astype(np.float)


if __name__ == '__main__':
    train_rate = 0.8
    data = load('../data/Iris.csv')
    np.random.shuffle(data)
    # 划分训练集和测试集
    train_num = int(data.shape[0]*train_rate)
    train_data_set = data[:train_num, :-1].T
    train_label = data[:train_num, -1].T
    test_data_set = data[train_num:, :-1]
    test_label = data[train_num:, -1]

    knn = Knn(train_data_set, train_label)
    predict_res = []
    for data, label in zip(test_data_set, test_label):
        res = knn.classify(data.T, 5)
        print('ground truth:', label, 'predict result:', res)
        predict_res.append(res[0][0])

    print('accuracy:', sum(np.ones((predict_res.__len__(), 1))[np.asarray(predict_res) == test_label])/test_label.shape[0])
