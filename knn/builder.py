"""
2019/1/10
构建kd树
以下所有 dataset 每列为一个特征向量
"""
import numpy as np
from knn.util import KdNode


class Builder:
    def __init__(self, data_set):
        self.data_set = data_set

    def _variance_r(self, data_set):
        """
        计算每个特征的方差，选择方差最大的特征作为划分dataset的纬度
        其实随机选也可以，不过这种方法构造的树好一点
        :param data_set: 每列是一个特征向量
        :return: r
        """
        # axis = 1是行， 0是列
        my_vars = np.var(data_set, axis=1, ddof=1)
        return np.argmax(my_vars)

    def _build(self, data_set_index):
        """
        构建kd-tree
        :param data_set_index: 用到的data set在 self.data_set中的 index,用self.data_set[:, data_set_index]获得data_set
        :return:
        """
        if data_set_index.size == 0:
            return None
        # 根据index获得data set
        data_set = self.data_set[:, data_set_index]
        # 计算方差，获得纬度
        r = self._variance_r(data_set)
        # 对data_set排序，获得index
        sorted_arg = np.argsort(data_set[r, :])
        # 得到节点的index(在self.data_set中)
        median_index = data_set_index[sorted_arg[int((len(sorted_arg)-1)/2)]]
        # 得到左右data set的index(在self.data_set中)，
        left = data_set_index[np.where(data_set[r, :] < self.data_set[r, median_index])]
        # 这里右边的计算方式是为了减少重复
        right = data_set_index[np.where(data_set[r, :] >= self.data_set[r, median_index])]
        equal_index = np.where(right == median_index)
        right = np.delete(right, equal_index[0][0])

        return KdNode(median_index, r, self._build(left), self._build(right))

    def build(self):
        """
        build主函数
        :return: kd-tree
        """
        init_index = np.linspace(0, self.data_set.shape[1], num=self.data_set.shape[1], endpoint=False, dtype=np.int64)
        return self._build(init_index)

