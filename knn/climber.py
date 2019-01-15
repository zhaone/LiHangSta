"""
2019/1/10
树搜索算法实现
"""
import numpy as np
from knn.util import BigTopHeap


class Climber:
    def __init__(self, data_set):
        self.data_set = data_set

    @staticmethod
    def _calc_dist(node1, node2):
        """
        :param node1: 节点1
        :param node2: 节点2
        :return: node1 node2之间的欧式距离（为了简化计算没开方）
        """
        sub = node1 - node2
        return np.dot(sub.T, sub)

    def _climb(self, node, v, heap):
        """
        :param node: 树当前节点
        :param v: 待分类点
        :param heap: 前k小的堆
        :return: 无，函数只是为了构建堆
        """
        if not node:
            return
        # 找出该维度v和node的值
        vr = v[node.axis]
        noder = self.data_set[node.axis, node.index]
        # 先遍左边
        if vr <= noder:
            self._climb(node.left, v, heap)
            if not heap.full() or (vr-noder)*(vr-noder) < heap.look()[1]:
                self._climb(node.right, v, heap)
        # 先遍右边
        elif vr > noder:
            self._climb(node.right, v, heap)
            if not heap.full() or (vr - noder) * (vr - noder) < heap.look()[1]:
                self._climb(node.left, v, heap)
        # 现在左右两棵树已经遍完了，判断该节点是否应该加入heap
        distance = self._calc_dist(v, self.data_set[:, node.index])
        if not heap.full() or distance < heap.look()[1]:
            heap.push(node.index, distance)

    def climb(self, v, k, kd_tree_root):
        """
        :param v: 待分类点
        :param k: KNN的参数k，为了构建堆
        :param kd_tree_root: kd tree的根节点
        :return:
        """
        BTHeap = BigTopHeap(k)
        self._climb(kd_tree_root, v, BTHeap)
        return BTHeap.get_all_index()
