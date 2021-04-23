import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
import altair as alt
import cv2
from altair_saver import save

def sigmoid(z):
	'''
	Function that returns the sigmnoid value of z
	'''
	return 1 / (1 + np.exp(-z))

def predict(X, w, b):
	'''
    It returns the value predicted by the model
    '''
	return np.round(sigmoid(np.dot(X, w)+ b))

def loss(y_pred,t):
	'''
    Function to calculate the loss
    '''
	return np.mean(-t*np.log(y_pred)-(1-t)*np.log(1-y_pred))


def report(y_pred, test):
	'''
    Function to calculate the accuracy and f-score
    '''

	tp = sum((test == 1) & (y_pred == 1))
	tn = sum((test == 0) & (y_pred == 0))
	fn = sum((test == 1) & (y_pred == 0))
	fp = sum((test == 0) & (y_pred == 1))

	accuracy = ((tp + tn)*100)/max(float(tn + tp + fn + fp), 1)
	precision = (tp)/max(float(tp + fp), 1)
	recall = (tp)/max(float(tp + fn), 1)
	fscore = (2*precision*recall)/(precision + recall)

	return accuracy, fscore


def graph(cost, lr):
	'''
    Function to plot graph of the learning curve vs iterations
    '''
	plt.plot(cost)
	plt.ylabel('cost')
	plt.xlabel('iterations')
	plt.title("Initialization : Uniform[0,1)")
	plt.show()

def stan(X):
	'''
    Function to standardize the data
    '''

	mean = np.mean(X, axis=0)			# axis = 0 for along the column
	std = np.std(X, axis=0, ddof=1)     #ddof = 1 for sample standard deviation
	X = X.astype(float)

	for i in range(0, X.shape[0]):
		for j in range(0, X.shape[1]):
			X[i][j] = (float(X[i][j] - mean[j])/float(std[j]))  #Xij = (Xij-mean)/Standard deviation

	return X


def plot_Decision_Boundary(X, y, W, b):

    fig = plt.figure()
    #X1 = []
    #X2 = []
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    for i in range(len(X)):
    	if y[i][0] == 0:
    		#X1.append([X[:2][0][i], X[:2][1][i]])
    		plt.scatter(X[i][0], X[i][1], marker = "o", c = "red")
    		if X[i][0] <= x_min:
    			x_min = X[i][0]
    		if X[i][0] >= x_max:
    			x_max = X[i][0]
    		if X[i][1] <= y_min:
    			y_min = X[i][1]
    		if X[i][1] >= y_max:
    			y_max = X[i][1]
    	elif y[i][0] == 1:
    		#X2.append([X[:2][0][i], X[:2][1][i]])
    		plt.scatter(X[i][0], X[i][1], marker = "o", c = "green")
    		if X[i][0] <= x_min:
    			x_min = X[i][0]
    		if X[i][0] >= x_max:
    			x_max = X[i][0]
    		if X[i][1] <= y_min:
    			y_min = X[i][1]
    		if X[i][1] >= y_max:
    			y_max = X[i][1]

    #print(X1[0])
    m = -W[0]/W[1]
    c = -b/W[1]
    #print(c)
    xmin, xmax = x_min, x_max
    ymin, ymax = y_min, y_max
    xd = X[:, 0]	#np.array([xmin, xmax])
    yd = m*xd + c + 82.5		# bias added due to standardization
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
    plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
    plt.xlabel('X0')
    plt.ylabel('X1')
    #plt.legend(['Admitted', 'Not admitted', 'line'])
    return 0


if __name__ == "__main__":

	data1 = load_breast_cancer(return_X_y=False, as_frame=True)
	#print(type(data1.data))
	data = data1.data
	target = data1.target
	#print(target)
	X = data.iloc[:, 0:2].values    	#Features
	#print(X[:, 0:2].shape)
	T = target.values 	 				#Result
	T = T.reshape(569, 1)


	# print(X)
	# print("------")
	# print(T)
	cost = []							#Initialize array to store the cost values at each iteration

	X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2)
	#model.fit(X_train, T_train)

	X_train_standard = stan(X_train)    #Standardise Training data
	X_test_standard = stan(X_test)      #Standardise Testing data

	epoch = 1000
	lr = 0.5
	np.random.seed(0)
	W = np.random.uniform(0,1,size=(X_train_standard.shape[1],1)) 	#intial weights
	#W = np.zeros((X_train_standard.shape[1],1))
	b = 1                                        			#bias

	for i in range(1,epoch+1):
		Z = np.dot(X_train_standard, W) + b
		y_predicted = sigmoid(Z)
		#print(y_predicted)
		error = loss(y_predicted, T_train)

		if epoch%50==0:
			print("Loss -------------> ",error)
			cost.append(error)

		if math.isnan(error):
			break
		
		grad = y_predicted - T_train
		gradient = np.dot(np.transpose(X_train_standard),grad)/(X_train_standard.shape[0])
		#print(gradient)
		grad_bias = np.average(grad)
		W = W - lr*gradient
		b = b - lr*grad_bias


	#print(W)
	#print(b)

	T_pred = predict(X_test_standard, W, b)

	acc, f = report(T_pred, T_test)
	print(" Test Accuracy = ", acc[0])
	print("F score = ", f[0])
	graph(cost, lr);

	#print(X[:, 0])
	#print(X[0][1])
	plot_Decision_Boundary(X, T, W, b)
	plt.show()