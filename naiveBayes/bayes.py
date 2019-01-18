# 贝叶斯主程序，这个是应用于连续型的
import numpy as np


# 极大似然估计求高斯分布
class Bayes:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.class_priority_possibility, self.all_conditional_possibility = self.calculate_priority_possibility()

    def calculate_gaussian_possibility(self, class_id, feature_axis):
        """
        计算class_id类的feature_axis的高斯分布函数
        :param class_id: 类
        :param feature_axis: 特征的索引
        :return: 该类该特征的高斯分布概率密度函数
        """
        index = self.label == class_id
        items = self.data[feature_axis, index]
        _mean = np.mean(items)
        _std = np.std(items)
        coefficient = 1/(_std*np.sqrt(2*np.pi))
        square_std = 2*_std*_std

        def gaussian_ck(x):
            print(x, _mean, _std, coefficient*np.exp(-np.square(x-_mean)/square_std))
            return coefficient*np.exp(-np.square(x-_mean)/square_std)
        return gaussian_ck

    def calculate_priority_possibility(self):
        """
        计算先验概率
        :return:
        class_priority_possibility: 类先验概率P(y_i)，采用极大似然估计
        all_conditional_possibility: x条ii按概率P(X^(j)=x^(j)|y_i)，正态分布，极大似然估计
        """
        all_class_id = np.unique(self.label)
        # 每个类每个特征的条件概率
        all_conditional_possibility = []
        # 类先验概率
        class_priority_possibility = []
        # 类id为_id
        for _id in all_class_id:
            # 计算类先验概率
            class_priority_possibility.append((_id, self.label[self.label == _id].shape[0]/self.label.shape[0]))
            # 该类对应的P(X^(j)=x^(j)|y_i)条件概率，类每个特征
            class_conditional_possibility = []
            for axis in range(self.data.shape[0]):
                class_conditional_possibility.append(self.calculate_gaussian_possibility(_id, axis))
            all_conditional_possibility.append(class_conditional_possibility)
        return class_priority_possibility, all_conditional_possibility

    def classify(self, sample):
        """
        分类
        :param sample: 要预测的特征向量
        :return: 类别
        """
        class_possibility = []
        _sum = 0
        for i in range(self.class_priority_possibility.__len__()):
            _id = self.class_priority_possibility[i][0]
            # 类先验概率
            possibility = self.class_priority_possibility[i][1]
            # 该类的所有条件概率函数
            class_conditional_possibility = self.all_conditional_possibility[i]
            for j in range(class_conditional_possibility.__len__()):
                possibility = possibility * class_conditional_possibility[j](sample[j])
            #  possibility 时 P(X=x|y=_id)
            _sum = _sum + possibility
            class_possibility.append((_id, possibility))

        res = []
        for item in class_possibility:
            res.append((item[0], item[1]/_sum))
        return res
