# Adaboost算法

> Adaboost算法是boosting集成学习中最为著名的算法之一，可分类可回归。
>
> Aadboost算法是模型为加法模型、损失函数为指数函数、学习算法为前向分步算法的学习方法。

## Adaboost算法流程

### 分类问题

**输入：**训练数据集$$T=\{(x_,y_1),(x_2,y_2), ...(x_m,y_m)\}$$，其中$x_i \in R^n$，$y_i\in \{-1,1\}$

**输出：**最终分类器$G(x)$

(1) 初始化训练数据的权值分布：

​				$$D(1) = (w_{11}, w_{12}, ...w_{1m}) ;\;\; w_{1i}=\frac{1}{m};\;\; i =1,2...m$$

(2) 对 k= 1,2,3,...,K:

​	(a) 使用具有权重分布的$D_k$数据集训练基本分类器$G_k$

​	(b) 计算$G_k$在训练数据集上的误差率：

​				$$e_k = P(G_k(x_i) \neq y_i) = \sum\limits_{i=1}^{m}w_{ki}I(G_k(x_i) \neq y_i)$$

​	(c) 计算$G_k$系数：

​				                        $\alpha_k = \frac{1}{2}log\frac{1-e_k}{e_k}$

​	(d) 更新训练数据集权值分布：

​					$D(k+1) = (w_{k+1,1}, w_{k+1,2}, ...w_{k+1,m})$

​					$w_{k+1,i} = \frac{w_{ki}}{Z_K}exp(-\alpha_ky_iG_k(x_i))$

​		这里$Z_k$是规范化因子$$Z_k = \sum\limits_{i=1}^{m}w_{ki}exp(-\alpha_ky_iG_k(x_i))$$，使得

(3) 构建基本分类为的线性组合：

​							$f(x) =\sum\limits_{k=1}^{K}\alpha_kG_k(x)$

得到最终分类器：

​							$$G(x) = sign(f(x))=sign(\sum\limits_{k=1}^{K}\alpha_kG_k(x))$$

### 回归问题

AdaBoost回归算法变种很多，下面的算法为Adaboost R2回归算法过程。

**输入：**训练数据集$$T=\{(x_,y_1),(x_2,y_2), ...(x_m,y_m)\}$$，其中$x_i \in R^n$，$y_i\in \{-1,1\}$

**输出：**最终回归器$G(x)$

(1) 初始化训练数据的权值分布：

​				$$D(1) = (w_{11}, w_{12}, ...w_{1m}) ;\;\; w_{1i}=\frac{1}{m};\;\; i =1,2...m$$

(2) 对 k= 1,2,3,...,K:

​	(a) 使用具有权重分布的$D_k$数据集训练基本分类器$G_k$

​	(b) 计算$G_k$在训练数据集上的最大误差：

​				$$E_k= max|y_i - G_k(x_i)|\;i=1,2...m$$

​	(c) 计算每个样本的相对误差：

​		如果是线性误差，则$e_{ki}= \frac{|y_i - G_k(x_i)|}{E_k}$

​		如果是平方误差，则$e_{ki}= \frac{(y_i - G_k(x_i))^2}{E_k^2}$

​		如果是指数误差，则$e_{ki}= 1 - exp(-\frac{y_i - G_k(x_i))}{E_k})$

​	(d) 计算回归误差率：$$e_k = \sum\limits_{i=1}^{m}w_{ki}e_{ki}$$

​	(e) 计算$G_k$系数：

​				                        $$\alpha_k =\frac{e_k}{1-e_k}$$

​	(f) 更新训练数据集权值分布：

​					$D(k+1) = (w_{k+1,1}, w_{k+1,2}, ...w_{k+1,m})$

​					$$w_{k+1,i} = \frac{w_{ki}}{Z_k}\alpha_k^{1-e_{ki}}$$

​		这里$Z_k$是规范化因子$$Z_k = \sum\limits_{i=1}^{m}w_{ki}\alpha_k^{1-e_{ki}}$$，使得

(3) 构建基本分类为的线性组合：

​							$$f(x) = \sum\limits_{k=1}^{K}(ln\frac{1}{\alpha_k})G_k(x)$$

得到最终分类器：

​							$$G(x) = f(x) = \sum\limits_{k=1}^{K}(ln\frac{1}{\alpha_k})G_k(x)$$

### Adaboost算法正则化

为了防止Adaboost过拟合，我们通常也会加入正则化项，这个正则化项我们通常称为步长(learning rate)。定义为$\nu$,对于前面的弱学习器的迭代$$f_{k}(x) = f_{k-1}(x) + \alpha_kG_k(x);$$

如果我们加上了正则化项，则有$$f_{k}(x) = f_{k-1}(x) + \nu\alpha_kG_k(x)$$

$\nu$的取值范围为$(0,1)$。对于同样的训练集学习效果，较小的$\nu$意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。

## Adaboost算法优缺点

一般来说，使用最广泛的Adaboost弱学习器是决策树和神经网络。对于决策树，Adaboost分类用了CART分类树，而Adaboost回归用了CART回归树。

> Adaboost的主要优点有：
>
> - Adaboost作为分类器时，分类精度很高
>
>
> - 在Adaboost的框架下，可以使用各种回归分类模型来构建弱学习器，非常灵活。
> - 作为简单的二元分类器时，构造简单，结果可理解。
> - 不容易发生过拟合
>
> Adaboost的主要缺点有：
>
> - **对异常样本敏感，异常样本在迭代中可能会获得较高的权重，影响最终的强学习器的预测准确性。**

## Adaboost推导

![](https://raw.githubusercontent.com/clhchtcjj/Pit-for-Typora/master/Adaboost.jpg)

## 参考文献

【1】 刘建平 [集成学习之Adaboost算法原理小结](http://www.cnblogs.com/pinard/p/6133937.html)

【2】李航 《统计学习方法》

【3】周志华《机器学习》



