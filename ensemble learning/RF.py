# -*- coding: utf-8 -*-
__author__ = 'CLH'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn import cross_validation,metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

'''
    整理自 刘建平 http://www.cnblogs.com/pinard/p/6143927.html
    sklearn.ensemble实践——Random Forest算法
    sklearn.ensemble类库中，有关RF的只有两个算法：RandomForestClassifier用于分类，RandomForestRegressor用于回归

'''

'''
    分类
    n_estimators: 也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。
    oob_score :即是否采用袋外样本来评估模型的好坏。默认识False
    criterion: 即CART树做划分时对特征的评价标准：基尼或信息增益
    min_impurity_split：节点划分最小不纯度，这个值限制了决策树的增长，如果某节点的不纯度(基于基尼系数，均方差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。一般不推荐改动默认值1e-7。
    bootstrap: 是否采用bootstrap采样法
'''

# 导入数据
train = pd.read_csv('train_modified.csv')
target = 'Disbursed' # 二分类的输出
IDcol = 'ID'
print(train['Disbursed'].value_counts())

# 构造样本特征
x_features = [x for x in train.columns if x not in [target,IDcol]]
X = train[x_features]
Y = train['Disbursed']

# 使用默认参数拟合
rf = RandomForestClassifier(oob_score=True,random_state=10)
rf.fit(X,Y)
print(rf.oob_score_)
y_predprob = rf.predict_proba(X)[:,1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(Y, y_predprob))

'''
    回归
    loss: GBDT算法中的损失函数。对于回归模型，有均方差"ls", 绝对损失"lad", Huber损失"huber"和分位数损失“quantile”。默认是均方差"ls"。一般来说，如果数据的噪音点不多，用默认的均方差"ls"比较好。如果是噪音点较多，则推荐用抗噪音的损失函数"huber"。而如果我们需要对训练集进行分段预测的时候，则采用“quantile”。
    alpha：当我们使用Huber损失"huber"和分位数损失“quantile”时，需要指定分位数的值。默认是0.9，如果噪音点较多，可以适当降低这个分位数的值。
'''

# 生成数据
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
Y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

rf = RandomForestRegressor(random_state=10)
rf.fit(X,Y)
Y_hat = rf.predict(X)
plt.figure()
# plt.subplot(1,2,1)
plt.scatter(X,Y,color='black')
plt.plot(X,Y_hat)
plt.title("Random Forest Regression")
plt.show()