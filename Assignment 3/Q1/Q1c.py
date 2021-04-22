import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from autograd import grad as GRAD
from sklearn.datasets import load_breast_cancer
#from sklearn import cross_validation
from sklearn.model_selection import cross_val_score

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
	#model.fit(X_train, T_train)

	X_train_standard = stan(X_train)    #Standardise Training data
	X_test_standard = stan(X_test)      #Standardise Testing data

	X_train0 = []
	X_train1 = []
	X_train2 = []
	T_train0 = []
	T_train1 = []
	T_train2 = []

	for i in range(len(X_train_standard)):
		if i%3 == 0:
			X_train0.append(X_train_standard[i])
			T_train0.append(T_train[i])
		elif i%3 == 1:
			X_train1.append(X_train_standard[i])
			T_train1.append(T_train[i])
		elif i%3 == 2:
			X_train2.append(X_train_standard[i])
			T_train2.append(T_train[i])



	epoch = 1000
	lr = 0.5
	np.random.seed(0)
	W = np.random.uniform(0,1,size=(X_train_standard.shape[1],1)) 	#intial weights
	#W = np.random.normal(0,1, size=(X_train_standard.shape[1],1))
	#W = np.zeros((X_train_standard.shape[1],1))
	b = 1                                        			#bias

	for i in range(1,epoch+1):
		# 2 = test
		x_train = []
		t_train = []
		x_test = np.array(X_train2)
		t_test = np.array(T_train2)
		accuracy = []

		for i in X_train0:
			x_train.append(i)
		for i in T_train0:
			t_train.append(i)
		for i in X_train1:
			x_train.append(i)
		for i in T_train1:
			t_train.append(i)

		x_train = np.array(x_train)
		t_train = np.array(t_train)

		Z = np.dot(x_train, W) + b
		y_predicted = sigmoid(Z)
		#print(y_predicted)
		error = loss(y_predicted, t_train)

		if epoch%50==0:
			print("Loss2 -------------> ",error)
			cost.append(error)

		if math.isnan(error):
			break

		grad = y_predicted - t_train
		gradient = np.dot(np.transpose(x_train),grad)/(x_train.shape[0])
		#gradient = GRAD(loss)
		grad_bias = np.average(grad)
		W = W - lr*gradient
		b = b - lr*grad_bias

		t_pred = predict(x_test, W, b)
		#scores = cross_val_score(T_pred, X, T, cv=3, scoring='accuracy')
		#print(scores)

		acc, f = report(t_pred, t_test)
		accuracy.append(acc)


		x_train = []
		t_train = []
		x_test = np.array(X_train1)
		t_test = np.array(T_train1)

		for i in X_train0:
			x_train.append(i)
		for i in T_train0:
			t_train.append(i)
		for i in X_train2:
			x_train.append(i)
		for i in T_train2:
			t_train.append(i)

		x_train = np.array(x_train)
		t_train = np.array(t_train)

		Z = np.dot(x_train, W) + b
		y_predicted = sigmoid(Z)
		#print(y_predicted)
		error = loss(y_predicted, t_train)

		if epoch%50==0:
			print("Loss1 -------------> ",error)
			cost.append(error)

		if math.isnan(error):
			break

		grad = y_predicted - t_train
		gradient = np.dot(np.transpose(x_train),grad)/(x_train.shape[0])
		#gradient = GRAD(loss)
		grad_bias = np.average(grad)
		W = W - lr*gradient
		b = b - lr*grad_bias
		t_pred = predict(x_test, W, b)
		#scores = cross_val_score(T_pred, X, T, cv=3, scoring='accuracy')
		#print(scores)

		acc, f = report(t_pred, t_test)
		accuracy.append(acc)

		x_train = []
		t_train = []
		x_test = np.array(X_train0)
		t_test = np.array(T_train0)

		for i in X_train2:
			x_train.append(i)
		for i in T_train2:
			t_train.append(i)
		for i in X_train1:
			x_train.append(i)
		for i in T_train1:
			t_train.append(i)

		x_train = np.array(x_train)
		t_train = np.array(t_train)

		Z = np.dot(x_train, W) + b
		y_predicted = sigmoid(Z)
		#print(y_predicted)
		error = loss(y_predicted, t_train)

		if epoch%50==0:
			print("Loss0 -------------> ",error)
			cost.append(error)

		if math.isnan(error):
			break

		grad = y_predicted - t_train
		gradient = np.dot(np.transpose(x_train),grad)/(x_train.shape[0])
		#gradient = GRAD(loss)
		grad_bias = np.average(grad)
		W = W - lr*gradient
		b = b - lr*grad_bias
		t_pred = predict(x_test, W, b)
		#scores = cross_val_score(T_pred, X, T, cv=3, scoring='accuracy')
		#print(scores)

		acc, f = report(t_pred, t_test)
		accuracy.append(acc)

		print("Cross validation fold accuracy ---->",accuracy)



		Z = np.dot(X_train_standard, W) + b
		y_predicted = sigmoid(Z)
		#print(y_predicted)
		error = loss(y_predicted, T_train)

		if epoch%50==0:
			print("Loss -------------> ",error)
			#print(accuracy)
			cost.append(error)

		if math.isnan(error):
			break

		
		#gradient = gradient_cal(y_predicted, T_train, X_train_standard)
		grad = y_predicted - T_train
		gradient = np.dot(np.transpose(X_train_standard),grad)/(X_train_standard.shape[0])
		#gradient = GRAD(loss)
		grad_bias = np.average(grad)
		W = W - lr*gradient
		b = b - lr*grad_bias


	#print(W)
	#print(b)

	T_pred = predict(X_test_standard, W, b)

	acc, f = report(T_pred, T_test)
	print("Overall Test Accuracy = ", acc[0])
	print("F score = ", f[0])
	graph(cost, lr);
