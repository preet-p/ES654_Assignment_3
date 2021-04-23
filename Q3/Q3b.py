import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
import autograd.numpy as np
from autograd import grad


def softmax(z):
    z -= np.max(z)
    return (np.exp(z).T / np.sum(np.exp(z), axis=1))

def predict(X, w, b):
	'''
    It returns the value predicted by the model
    '''
	return np.argmax(softmax(np.dot(X, w)+ b))


def cal_entropy_loss(W, b, T_train, X_train):
	Z = np.dot(X_train, W) + b
	y_predicted = softmax(Z)
	#print("------")
	#print(W)
	#print(y_predicted)

	return -1 * np.mean(T_train * np.log(y_predicted))

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


if __name__ == "__main__":

	data1 = load_digits(return_X_y=False, as_frame=True)
	#print(type(data1.data))
	data = data1.data
	target = data1.target
	#print(target)
	X = data.iloc[:, 0:65].values    	#Features
	T = target.values 	 				#Result
	T = T.reshape(1797, 1)

	# print(X)
	# print("------")
	# print(T)
	cost = []							#Initialize array to store the cost values at each iteration

	X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2)


	epoch = 5000
	lr = 0.0001
	np.random.seed(0)
	#W = np.random.uniform(0,1,size=(X_train.shape[1],1)) 	#intial weights
	W = np.zeros((X_train.shape[1],1))
	b = 1                                        			#bias

	gradient = grad(cal_entropy_loss)

	for i in range(1,epoch+1):
		Z = np.dot(X_train, W) + b
		y_predicted = softmax(Z)
		#print(y_predicted)
		error = cal_entropy_loss(W, b, T_train, X_train)

		if epoch%100==0:
			print("Loss -------------> ",error)
			cost.append(error)

		if math.isnan(error):
			break


		gra = y_predicted - T_train
		#print(gradient(W, b, T_train, X_train))
		grad_bias = np.average(gra)
		W = W - lr*gradient(W, b, T_train, X_train)
		#print(W)
		b = b - lr*grad_bias


	#print(W)
	#print(b)

	T_pred = predict(X_test, W, b)

	acc, f = report(T_pred, T_test)
	print(" Test Accuracy = ", acc[0])
	print("F score = ", f[0])
	graph(cost, lr);
