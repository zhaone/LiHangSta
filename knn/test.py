"""
测试代码
"""
from knn.util import BigTopHeap
from knn.climber import Climber
import numpy as np

def test_BTHeap():
    BTHeap = BigTopHeap(3)
    a = BTHeap.full()
    c = BTHeap.look()
    BTHeap.push(1, 3.9)
    BTHeap.push(1, 4.5)
    a = BTHeap.full()
    c = BTHeap.look()
    BTHeap.push(1, 4.0)
    BTHeap.push(1, 4.3)
    a = BTHeap.full()
    c = BTHeap.look()


def test_calc_distance():
    a = Climber._calc_dist(np.ones((4, 1)), np.ones((4, 1))*3)
    print('kkk')


if __name__ == '__main__':
    # test_BTHeap()  # 这个大顶堆测试过，是对的
    test_calc_distance()  # 这个计算距离是对的
