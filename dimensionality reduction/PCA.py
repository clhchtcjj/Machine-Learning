#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'


'''
    关于sklearn降维算法，参见http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition

'''

'''
    PCA算法
    参数介绍
    n_components：降维后的数据维度（特征数）
                  （1）默认值min(n_samples, n_features)
                  （2）自己指定维度
                  （3）如果n_components == ‘mle’ 并且 svd_solver == ‘full’用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维
                  （4）如果0 < n_components < 1 并且svd_solver == ‘full’ 根据样本特征方差来决定降维到的维度数

    whiten：是否进行白化：对降维后的数据的每一个特征进行归一化，方差为1
    svd_solver：指定奇异值分解svd的方法 'auto': 默认的，模型自动选择；‘full’:传统意义的SVD
                arpack和randomized的适用大数据场景，它使用了一些加快SVD的随机算法
    copy : 默认True,如果设置为False，会改变原始数据

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA,KernelPCA

# 生成数据
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇

X,y = make_blobs(n_samples=100000, n_features=3, centers=[[3,3,3],[0,0,0],[1,1,1],[2,2,2]],cluster_std=[0.2,0.1,0.2,0.2],random_state=9)

fig = plt.figure()
ax = Axes3D(fig,rect=[0,0,1,1],elev=30,azim=20)
plt.scatter(X[:,0],X[:,1],X[:,2],marker='+')
plt.show()

# 直接投影不降维
pca = PCA(n_components=3)
pca.fit(X)
# 观察每个部分的方差比例和总数
print(pca.explained_variance_ratio_,pca.explained_variance_)

# 降维 3=》2
pca = PCA(n_components=2)
pca.fit(X)
# 观察每个部分的方差比例和总数
print(pca.explained_variance_ratio_,pca.explained_variance_)

# 获得重构后的数据
X_hat = pca.transform(X)
plt.scatter(X_hat[:,0],X_hat[:,1],marker='o')
plt.show()

# 降维 3=》1
pca = PCA(n_components='mle',svd_solver='full') # 需要同时设置svd_solver参数
pca.fit(X)
# 观察每个部分的方差比例和总数
print(pca.n_components_,pca.explained_variance_ratio_,pca.explained_variance_)


'''
    PCA算法
    参数介绍
    n_components：降维后的数据维度（特征数）
    kernel : 核函数，“linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
    gamma : rbf, poly and sigmoid kernels 的一个参数设置
    eigen_solver : [‘auto’|’dense’|’arpack’] 特征值分解方法
    fit_inverse_transform：学习逆变换
'''

# 构造数据
# X为样本特征，Y为样本簇类别， 共400个样本,y为样本所属类别
X, y = make_circles(n_samples=400,factor=0.3,noise=.05)

kpca = KernelPCA(kernel='rbf',fit_inverse_transform=True,gamma=10)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)

plt.figure()
plt.subplot(2, 2, 1, aspect='equal')
plt.title("Original space")
reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c="pink", s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="yellow",s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")


plt.subplot(2, 2, 2, aspect='equal')
plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="pink", s=20, edgecolor='k')
plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="yellow",s=20, edgecolor='k')
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.subplot(2, 2, 3, aspect='equal')
plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="pink", s=20, edgecolor='k')
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="yellow", s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel("1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.subplot(2, 2, 4, aspect='equal')
plt.scatter(X_back[reds, 0], X_back[reds, 1], c="pink",s=20, edgecolor='k')
plt.scatter(X_back[blues, 0], X_back[blues, 1], c="yellow",s=20, edgecolor='k')
plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)
plt.show()

print("总结：KPCA可以用于非线性可分的情况")