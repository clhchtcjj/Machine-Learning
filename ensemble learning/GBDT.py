# -*- coding: utf-8 -*-
__author__ = 'CLH'

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn import cross_validation,metrics
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
'''
    整理自 刘建平 http://www.cnblogs.com/pinard/p/6143927.html
    sklearn.ensemble实践——GBDT算法
    sklearn.ensemble类库中，有关GBDT的只有两个算法：GradientBoostingClassifier用于分类，GradientBoostingRegressor用于回归
'''

'''
    分类
    loss: 即我们GBDT算法中的损失函数。对于分类模型，有对数似然损失函数"deviance"和指数损失函数"exponential"两者输入选择。默认是对数似然损失函数"deviance"。
    n_estimators: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数
    learning_rate: 即每个弱学习器的权重缩减系数ν，也称作步长
    subsample: 子采样，取值为(0,1]，注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。如果取值为1，则全部样本都使用，等于没有使用子采样。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐在[0.5, 0.8]之间，默认是1.0，即不使用子采样。
    init: 即我们的初始化的时候的弱学习器
    其他的参数在Adaboost介绍过了
'''

# # 导入数据
# train = pd.read_csv('train_modified.csv')
# target = 'Disbursed' # 二分类的输出
# IDcol = 'ID'
# print(train['Disbursed'].value_counts())
#
# # 构造样本特征
# x_features = [x for x in train.columns if x not in [target,IDcol]]
# X = train[x_features]
# Y = train['Disbursed']
#
# # 使用默认参数拟合
# gbm0 = GradientBoostingClassifier(random_state=10)
# gbm0.fit(X,Y)
# y_pred = gbm0.predict(X)
# y_predprob = gbm0.predict_proba(X)[:,1]
# print("Accuracy : %.4g" % metrics.accuracy_score(Y.values, y_pred))
# print("AUC Score (Train): %f" % metrics.roc_auc_score(Y.values, y_predprob))
#
# # 调参过程
# # 1 步长 和 迭代次数
# param_test1 = {'n_estimators': range(20,81,10)}
# gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1,min_samples_split=300,
#                                                              min_samples_leaf=20,max_depth=8,max_features='sqrt',
#                                                              subsample=0.8,random_state=10),
#                         param_grid=param_test1,scoring='roc_auc', iid=False,cv=5)
# gsearch1.fit(X,Y)
# print(gsearch1.best_params_, gsearch1.best_score_)
#
# # 通过上面的实验，发现最好的迭代次数是60
#
# # 2 max_depth 和 min_samples_split 进行网格搜索
# param_test2 = {'max_depth': range(3,14,2),'min_samples_split':range(100,801,200)}
# gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1,n_estimators=60,
#                                                              min_samples_leaf=20,max_features='sqrt',
#                                                              subsample=0.8,random_state=10),
#                         param_grid=param_test2,scoring='roc_auc', iid=False,cv=5)
# gsearch2.fit(X,Y)
# print(gsearch2.best_params_, gsearch2.best_score_)
#
# # 通过上面的实验，发现好的最大深度是7
# # min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。
# # 下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
#
# param_test3 = {'min_samples_leaf': range(60,101,10),'min_samples_split':range(100,1900,200)}
# gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1,n_estimators=60,max_depth=7,
#                                                              max_features='sqrt',
#                                                              subsample=0.8,random_state=10),
#                         param_grid=param_test3,scoring='roc_auc', iid=False,cv=5)
# gsearch3.fit(X,Y)
# print(gsearch3.best_params_, gsearch3.best_score_)


# 下面可以用类似的方法来对其他参数进行调整...此处不再赘述

'''
    回归
    loss: GBDT算法中的损失函数。对于回归模型，有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”。默认是均方差"ls"。一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好。如果是噪音点较多，则推荐用抗噪音的损失函数"huber"。而如果我们需要对训练集进行分段预测的时候，则采用“quantile”。
    alpha：当我们使用Huber损失"huber"和分位数损失“quantile”时，需要指定分位数的值。默认是0.9，如果噪音点较多，可以适当降低这个分位数的值。
'''

# 生成数据
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
Y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

gbdt = GradientBoostingRegressor(random_state=10)
gbdt.fit(X,Y)
Y_hat = gbdt.predict(X)
plt.figure()
# plt.subplot(1,2,1)
plt.scatter(X,Y,color='black')
plt.plot(X,Y_hat)
plt.title("GBDT Regression")
plt.show()