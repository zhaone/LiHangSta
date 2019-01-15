"""
2019/1/10
定义用到的大顶堆和节点类
"""
import heapq as hq  # https://docs.python.org/3.0/library/heapq.html


class KdNode:
    """
    节点类
    :mem
    index 节点位置
    axis 划分纬度
    left 左孩子位置
    right 右孩子位置
    """
    def __init__(self, index, axis, left, right):
        self.index = index
        self.axis = axis
        self.left = left
        self.right = right


class BigTopHeap:
    """
    保存距离前k小的队列
    队列元素shape为(index, -distance)
    之所以是-distance，是因为python只有小顶堆，而我们要的大顶堆
    """
    def __init__(self, k):
        """
        :param k: 堆大小
        """
        if k < 1:
            raise Exception(self, 'k 必须大于等于1！')
        self.maxsize = k
        self.heap = []

    def look(self):
        """
        查看堆顶元素，即最大元素
        :return: 堆顶元素
        """
        if len(self.heap) == 0:
            return None
        top = self.heap[0]
        top = (top[0], -top[1])
        return top

    def push(self, index, distance):
        """
        将节点加入队列
        :param index: 节点的index
        :param distance: 节点到分类点的距离
        :return:
        """
        if len(self.heap) < self.maxsize:
            hq.heappush(self.heap, (index, -distance))
        else:
            hq.heapreplace(self.heap, (index, -distance))

    def full(self):
        """
        :return: 堆是否已满
        """
        return self.heap.__len__() == self.maxsize

    def get_all_index(self):
        """
        :return: 堆中所有元素的index
        """
        return [x[0] for x in self.heap]
