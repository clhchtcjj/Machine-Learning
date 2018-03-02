
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression



def rssError(y,yHat):
	return ((y-yHat)**2).sum()

def regularize(X):
	inMat = X.copy()
	inMeans = np.mean(inMat,0)
	inVar = np.var(inMat,0)
	inMat = (inMat-inMeans)/inVar
	return inMat


def stageWise(x,y,eps=0.01,numIter=100):
	X = np.mat(x).T
	Y = np.mat(y).T
	yMean = np.mean(Y,0)
	X = regularize(X)
	m,n = np.shape(X)
	returnMat = np.zeros((numIter,n))
	ws = np.zeros(n,1));wsTest = ws.copy(); wsMax = ws.copy()
	for i in range(numIter):
		lowestError = np.inf
		for j in range(n):
			for sign in [-1,1]:
				wsTest = ws.copy()
				wsTest[j] += eps*sign
				yTest = X * wsTest
				rssE = rssError(Y.A,yTest.A)
				if rssE < lowestError:
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy
		returnMat[i,:] = ws.T
	return returnMat

