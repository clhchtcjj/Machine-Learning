
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
w,b = call_sklearn(x,y)

line_x = [round(min(x))-1, round(max(x))+1]
line_y = [w*x_i + b for x_i in line_x]

plt.figure(figsize=(8,8))
plt.scatter(x,y)
plt.plot(line_x,line_y,color = 'red', lw = '2')

plt.ylabel('y')
plt.xlabel('x')
plt.title('olsr')

ftext = 'y = wx + b = {:.3f}x + {:.3f}'.format(w,b)
plt.figtext(.15,.8,ftext,fontsize=11,ha='left')

plt.show()
