import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
# from jax import grad as GRAD
import autograd.numpy as np
from autograd import grad

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

def cal_loss(W, b, T_train, X_train_standard):
	Z = np.dot(X_train_standard, W) + b
	y_predicted = sigmoid(Z)
	#print("------")
	#print(W)
	#print(y_predicted)

	return np.mean(-T_train*np.log(y_predicted)-(1-T_train)*np.log(1-y_predicted))

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


if __name__ == "__main__":
	#loss_cal = GRAD(loss)

	data1 = load_breast_cancer(return_X_y=False, as_frame=True)
	#print(type(data1.data))
	#print(data1.data.head)
	#data = pd.read_csv('./data_banknote_authentication.csv')
	data = data1.data
	target = data1.target
	#print(target)
	X = data.iloc[:, 0:30].values    	#Features
	#T = data.iloc[:, 4:5].values
	T = target.values 	 	#Result
	T = T.reshape(569, 1)

	#print(ST)

	# print(X)
	# print("------")
	# print(T)
	cost = []							#Initialize array to store the cost values at each iteration

	X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2)

	X_train_standard = stan(X_train)    #Standardise Training data
	X_test_standard = stan(X_test)      #Standardise Testing data

	epoch = 5000
	lr = 0.75
	L1_coeff = 0.0002
	L2_coeff = 0.000001
	np.random.seed(0)
	W = np.zeros((X_train_standard.shape[1],1))
	b = 1                                     			#bias

	gradient = grad(cal_loss)

	for i in range(1,epoch+1):
		Z = np.dot(X_train_standard, W) + b
		y_predicted = sigmoid(Z)

		error = cal_loss(W, b, T_train, X_train_standard)
		error = error + L1_coeff*np.sign(np.average(W + b))
		#error = error + L2_coeff*np.square(np.average(W + b))
		#print(y_predicted, T_train)

		if epoch%100==0:
			print("Loss -------------> ",error)
			cost.append(error)

		if math.isnan(error):
			#print("error", error)
			break

		
		#print(gradient(y_predicted, T_train))
		#gradient = gradient_cal(y_predicted, T_train, X_train_standard)
		gra = y_predicted - T_train
		#print(gradient(W, b, T_train, X_train_standard))
		#gradient = np.dot(np.transpose(X_train_standard),grad)/(X_train_standard.shape[0])
		#gradient = GRAD(loss)
		grad_bias = np.average(gra)
		W = W - lr*gradient(W, b, T_train, X_train_standard) + L1_coeff*np.sign(W)
		#print(W)
		b = b - lr*grad_bias + L1_coeff*np.sign(b)

		# W = W - lr*gradient(W, b, T_train, X_train_standard) + L2_coeff*2*(W)
		# b = b - lr*grad_bias + L2_coeff*2*(b)


	#print(W)
	#print(b)

	T_pred = predict(X_test_standard, W, b)
	# scores = cross_val_score(T_pred, X, T, cv=3, scoring='accuracy')
	# print(scores)

	acc, f = report(T_pred, T_test)
	print(" Test Accuracy = ", acc[0])
	print("F score = ", f[0])
	graph(cost, lr);
