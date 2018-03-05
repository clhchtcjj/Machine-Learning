# -*- coding: utf-8 -*-
__author__ = 'CLH'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets.samples_generator import make_regression
'''
    整理自 刘建平 http://www.cnblogs.com/pinard/p/6136914.html
    sklearn.ensemble实践——Adaboost算法
    sklearn.ensemble类库中，有关Adaboost的只有两个算法：AdaBoostClassifier用于分类，AdaBoostRegressor用于回归
'''

'''
    分类
    AdaBoostClassifier参数介绍
    base_estimator：弱学习器，需要支持带权重的样本，常用的有决策树（CART、默认设定）、神经网络（MLP）
    algorithm：分类器中有两种算法来度量弱学习器的权重：SAMME和SAMME.R（默认设定，因为迭代快，但是需要弱学习器是支持概率预测的分类器）。前者按照分类结果，后者按照预测概率
    n_estimators：弱学习器的最大迭代次数，或者说最大的弱学习器的个数。太小容易欠拟合，太多容易过拟合。默认50
    learning_rate:正则项系数，较小的ν意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的开始调参，默认是1。

    DecisionTreeClassifier参数介绍
    max_features：划分时考虑的最大特征数，一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。
    min_samples_split：内部节点再划分所需最小样本数，这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
    min_samples_leaf：叶子节点最少样本数， 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。 默认是1,可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
    min_weight_fraction_leaf：叶子节点最小的样本权重和，这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝。 默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。
    max_leaf_nodes：最大叶子节点数，通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制，具体的值可以通过交叉验证得到。
'''

# 生成数据
# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
X1, Y1 = make_gaussian_quantiles(cov=2,n_samples=500,n_features=2,n_classes=2,random_state=1)

# 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为1.5
X2, Y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,n_samples=400, n_features=2, n_classes=2, random_state=1)

# 将两组数据合成一组数据
# print(X1,X2,Y1,Y2)
X = np.concatenate((X1,X2))
Y = np.concatenate((Y1,-Y2+1))
plt.figure()
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],s=10,marker='o',c=Y)

# 基于决策树进行分类拟合
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME", n_estimators=200, learning_rate=0.8)

bdt.fit(X,Y)

# 构造数据网格，以0.02为间距
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = bdt.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(1,2,2)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)
plt.title("Boosted Decision Tree Classifier")
plt.show()

print("Score:", bdt.score(X,Y))


'''
    回归
    与分类相比，多了一个参数loss：这个参数只有AdaBoostRegressor有，Adaboost.R2算法需要用到。有线性‘linear’, 平方‘square’和指数 ‘exponential’三种选择, 默认是线性，一般使用线性就足够了，除非你怀疑这个参数导致拟合程度不好。

'''

# 生成数据
# X,Y,coef = make_regression(n_samples=1000,n_features=1,noise=10,coef=True)

# 另一组非线性数据
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
Y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])


plt.figure()
# plt.subplot(1,2,1)
plt.scatter(X,Y,color='black')
# plt.plot(X,X*coef,color="blue",linewidth = 3)
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300)

regr.fit(X,Y)
Y_hat = regr.predict(X)
plt.plot(X,Y_hat)
plt.title("Boosted Decision Tree Regression")
plt.show()
print("Score",regr.score(X,Y))


