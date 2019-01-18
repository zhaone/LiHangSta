import numpy as np
from naiveBayes.pretreatment import split_train_test
from naiveBayes.bayes import Bayes


def load(path, shuffle=True):
    """
    加载数据
    :param path: 数据集路径
    :param shuffle: 是否shuffle
    :return:
    SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
    数据类型为np.float，每列一个特征向量
    返回了所有的值，如果想取某类或者某个特征，自己取就OK了
    """
    tmp = np.loadtxt(path, dtype=np.str, delimiter=',')
    tmp = tmp[1:, :]
    tmp = tmp[:, 1:]
    # 三种花类别分别用 0 1 2表示
    tmp[np.argwhere(tmp[:, -1] == 'Iris-setosa'), -1] = '0'
    tmp[np.argwhere(tmp[:, -1] == 'Iris-versicolor'), -1] = '1'
    tmp[np.argwhere(tmp[:, -1] == 'Iris-virginica'), -1] = '2'
    if shuffle:
        np.random.shuffle(tmp)
    _data = tmp[:, :-1].astype(np.float)
    _label = tmp[:, -1].astype(np.float)
    return _data.T, _label.T


if __name__ == '__main__':
    data, label = load('../data/Iris.csv', shuffle=True)
    train_data, test_data, train_label, test_label = split_train_test(data, label)
    bayes = Bayes(train_data, train_label)
    for _sample, _label in zip(test_data.T, test_label):
        print(bayes.classify(_sample), _label)
