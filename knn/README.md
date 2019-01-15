KNN算法实现-基于KD树
=======================
> 算法理论参考：[kd 树算法之详细篇-宏观经济算命椰](https://zhuanlan.zhihu.com/p/23966698)
#### 代码结构
```
builder.py              // 创建kd树
climber.py              // kd树搜索算法
KNearestNeighbor.py     // knn类，集合build和classify
main.py                 // 主程序，对鸢尾花数据集进行分类
README.md
test.py                 // 对有关数据结构进行测试
util.py                 // 大顶堆，kd树节点定义
__init__.py
```
代码具体原理见代码中注释。