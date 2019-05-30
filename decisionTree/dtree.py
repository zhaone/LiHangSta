import numpy as np
import pandas as pd
from decisionTree.process import  do_precess
from decisionTree.util import gini

#--------------------------------------分类-------------------------------------------------------------
def classify(tree, feature):
    if isinstance(tree, tuple):
        return tree[0]
    key = list(tree.keys())[0]
    feaVal = feature[key]
    return classify(tree[key][feaVal], feature)
#--------------------------------------剪枝-----------------------------------------------------------
def deleteChild(root):
    optKey = None
    optVal = None
    alpha = 1
    def calcGt(parentKey, parentValue, root):
        #  如果该树没有子树（数据结构为tuple)
        #        leafNum=1，类分布为[1数目，类2数目]， C(t)为基尼
        if isinstance(root, tuple):
            return 1, root[1], gini(root[1])
        #  如果该树有子树（数据结构为dictionary)
        childrenClass = []
        childredGTt = []
        leafNum = 0
        for key, values in root.items():
            for value in values:
                #  对根的子树调用calcGt，获得子树所有的leafNum, 所有leaf的类分布，子树的C(T_t)
                child = root[key][value]
                childLeafNum, childClass, childGTt = calcGt(key, value, child)
                leafNum = leafNum + childLeafNum
                childrenClass.append(childClass)
                childredGTt.append(childGTt)
        #  计算这个树的C(t), C(T_t), g(t)
        # 计算C(t)
        rootClass = np.sum(childrenClass, axis=0)
        Ct = gini(rootClass / rootClass.sum())
        # 计算C(T_t)
        prob = np.sum(childrenClass, axis=1) / rootClass.sum()
        CTt = np.dot(prob * np.asarray(childredGTt).T)
        # 计算g(t)
        gt = (Ct - CTt) / leafNum
        if gt < alpha:
            alpha = gt
            optKey = parentKey
            optVal = parentValue
        return leafNum, rootClass, CTt
    # 返回root删除（如果有的话）应该删除的节点后的树
    def innerDelete(root, akey, avalue):
        if isinstance(tree, tuple):
            return tree
        for key, values in root.items():
            for value in values:
                if key == akey and value == avalue:
                    _, classDist, _ = calcGt(None, None, root)
                    root[key][value] = (0 if classDist[0] > classDist[1] else 1, classDist)
                    return root
                else:
                    root[key][value] = innerDelete(root[key][value], akey, avalue)
        return root

    # TODO 给一个树，计算这个树的C(t), C(T_t), g(t)，对应李航第(3)
    # 树，以及树的根节点
    # 对根结点递归调用caltGt(t)，记下最小的g(t) 以及对应的树节点
    calcGt(None, None, root) #这一句对alpha optKey optVal赋值
    #  todo 删除树的一个节点,对应李航第(4)
    # 删除g(t)对应的树节点，修改分类
    # 返回删除节点后的新树和g(t)
    return innerDelete(root, optKey, optVal), alpha

class Dtree:
    def __init__(self, data, labels, evaFunc):
        self.train = data.iloc[1:700, :]
        self.labels = labels.iloc[1:700, :]
        self.test = data.iloc[700:800, :]
        self.testLabel = labels.iloc[700:800, :]
        self.ef = evaFunc
        # 提取所有特征
        self.uniFea = {}
    # 计算哪个label出现的次数最多
#--------------------------------------生成树-------------------------------------------------------
    def _countClass(self, dataIndex):
        labels = self.labels[dataIndex]
        return labels.value_counts(True)

    def _splitData(self, data, dataIndex):
        # 最大的基尼指数
        minGini = 1.1
        # 最优的key和value
        optKey = None
        # 遍历每个(key,value)对
        for key, values in self.uniFea.items():
            giniVal = 0
            expects = data[key].value_counts(True)
            fea = np.asarray(data[key].tolist())
            for value in values:
                index = dataIndex[np.where(fea == value)]
                if len(index) == 0:
                    continue
                prob = self.labels[index].value_counts(True)
                giniVal = giniVal + self.ef(prob.tolist())*expects[value]
            if giniVal!=0 and giniVal < minGini:
                minGini =giniVal
                optKey = key
        return optKey, minGini
    # 递归函数二
    def _doGrow(self, dataIndex):
        node = {}
        if len(dataIndex) == 0:
            return None

        counts = self.labels[dataIndex].value_counts()
        class0Num = 0 if not 0 in counts.index else counts[0]
        class1Num = 0 if not 1 in counts.index else counts[1]
        classDist = {0: class0Num, 1: class1Num}

        data = self.train.iloc[dataIndex]
        # 选择最优的切分点
        key, minGini = self._splitData(data, dataIndex)
        # 说明所有key, value用完了
        if not key:
            return (counts.index[0], classDist)
        # 说明基本是一个类
        if minGini < 0.1:
            return (counts.index[0], classDist)
        # 可以继续往下分
        fea = np.asarray(data[key].tolist())
        node[key] = {}
        for value in self.uniFea[key]:
            index = dataIndex[np.where(fea == value)]
            # 说明使用最好的也分不动了
            if len(index) == len(dataIndex):
                return (counts.index[0], classDist)
            # 继续分
            if len(index) == 0:
                continue
            node[key][value] = self._doGrow(index)
        return node
    # 建树的主函数
    def grow(self):
        for colName in self.train.columns:
            self.uniFea[colName] = self.train[colName].unique()
        allIndex = [i for i in range(len(train))]
        allIndex = np.asarray(allIndex)
        return self._doGrow(allIndex)

    def _cut(self, root):
        #  todo 删除树的一个节点,对应李航第全部
        alphas = [0]
        trees = [root]
        node = root
        # 循环调用deleteChild获得树序列和g(t)
        while True:
            if isinstance(root, tuple):
                break
            node, alpha = deleteChild(node)
            trees.append(node)
            alphas.append(alpha)
        # 对树序列调用classify，交叉验证获得最好的树
        bestTree = 0
        bestAccuracy = 0
        for tree in trees:
            correct = 0
            for i in range(self.test.length):
                if self.testLabel == classify(tree, test.iloc[i]):
                    correct = correct + 1
            accuracy = correct/self.test.length
            if accuracy > bestAccuracy:
                bestTree = tree
        # 返回最好的树
        return bestTree, bestAccuracy


if __name__ == '__main__':
    train = pd.read_csv('../data/titanic/train.csv')
    test = pd.read_csv('../data/titanic/test.csv')

    train = do_precess(train)
    test = do_precess(test)

    labels = train['Survived']
    features = train.drop(['Survived'], axis=1)

    dtree = Dtree(features, labels, gini)
    tree = dtree.grow()
    print(tree)
