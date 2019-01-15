"""
2019/1/10
Knn 分类器定义
"""
from knn.builder import Builder
from knn.climber import Climber
from collections import Counter


class Knn:
    def __init__(self, data_set, classes):
        self.classes = classes
        builder = Builder(data_set)
        self.kd_tree = builder.build()
        self.climber = Climber(data_set)

    def classify(self, v, k):
        k_nears_index = self.climber.climb(v, k, self.kd_tree)
        k_nears_classes = self.classes[k_nears_index]
        counter = Counter(k_nears_classes)
        return counter.most_common(1)
