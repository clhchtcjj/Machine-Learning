
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression



def matrix_olsr(x,y):
	X = np.vstack([x,np.ones(len(x))]).T
	return (np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y)


def classic_olsr(x,y):
	m = len(x)
	x_avg = sum(x)/m
	y_avg = sum(y)/m
	var_x, cov_xy = 0, 0
	for x_i,y_i in zip(x,y):
		temp = x_i - x_avg
		var_x += temp ** 2
		cov_xy += temp * (y_i - y_avg)
	w = cov_xy*1.0 / var_x
	b = y_avg - w * x_avg
	return (w,b)

def call_sklearn(x,y):
	lr = LinearRegression()
	x_ = np.matrix(x).T
	y_ = np.matrix(y).T
	print x_.shape,y_.shape
	lr.fit(x_,y_)
	print (lr.coef_,lr.intercept_)
	return (lr.coef_[0,0],lr.intercept_[0])


def point_lwlr(point,x,y,k):
	X = np.vstack([x,np.ones(len(x))]).T
	Y = np.mat(y).T
	m = len(x)
	L = np.mat(np.eye((m)))
	for i in range(m):
		diffMat = np.mat([point,1] - X[i,:])
		# print m,point,diffMat
		L[i,i] = np.exp(diffMat*diffMat.T/(-2.0 * k ** 2))
	temp = X.T * (L * X)
	if np.linalg.det(temp) == 0.0:
		print("This matrix is singular, can not do inverse")
		return 
	# print temp.shape,X.shape,L.shape,Y.shape,(L.dot(Y)).shape
	W = temp.I * X.T * L * Y
	# print (np.mat([point,1]) * W)[0,0]
	return (np.mat([point,1]) * W)[0,0]

def lwlr(x,y,k):
	m = len(x)
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = point_lwlr(x[i],x,y,k)
	return yHat




'''
visualization
'''

random.seed(12345)

'''
construct data
'''
x = [x_i * random.randrange(8,12)/10 for x_i in range(500)]
y = [y_i * random.randrange(8,12)/10 for y_i in range(100,600)]

# w,b = matrix_olsr(x,y)
# w,b = classic_olsr(x,y)
# w,b = call_sklearn(x,y)

# line_x = [round(min(x))-1, round(max(x))+1]
# line_y = [w*x_i + b for x_i in line_x]


# plt.figure(figsize=(8,8))
# plt.scatter(x,y)
# plt.plot(line_x,line_y,color = 'red', lw = '2')

# plt.ylabel('y')
# plt.xlabel('x')
# plt.title('olsr')

# ftext = 'y = wx + b = {:.3f}x + {:.3f}'.format(w,b)
# plt.figtext(.15,.8,ftext,fontsize=11,ha='left')

# plt.show()

k = 1
X = np.mat(x).T
Y = np.mat(y).T
YHat = lwlr(x,y,k)

srtInd = X[:,0].argsort(0)
xSort = X[srtInd][:,0,:]


# print X,srtInd
plt.figure(figsize=(8,8))
plt.scatter(x,y)
# print xSort[:,0],YHat[srtInd]
plt.plot(xSort[:,0],YHat[srtInd],color = 'red', lw = '2')

plt.ylabel('y')
plt.xlabel('x')
plt.title('lwlr')

# ftext = 'y = wx + b = {:.3f}x + {:.3f}'.format(w,b)
# plt.figtext(.15,.8,ftext,fontsize=11,ha='left')

plt.show()
