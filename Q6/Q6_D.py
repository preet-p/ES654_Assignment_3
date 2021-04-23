import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

def get_data():

    data1 = load_digits(return_X_y=False)
    #print(type(data1.data))
    X = data1.data                      #Features
    T = data1.target                    #Result
    T = T.reshape(1797, 1)

    data = []
    for i in range(X.shape[0]):
        a = []
        for j in range(X.shape[1]):
            a.append(float(X[i][j]))
        b = []
        for j in range(10):
            b.append(0)
        b[T[i][0]] = 1
        b = np.array(b)
        b = b.reshape(10,1)
        a.append(b)
        data.append(a)

    return normalize_data(data)

def normalize_data(data):
    '''
    Normalizes the dataset
    X[i] = X[i] - min(ith column) / (max(ith column) - min(ith column))
    '''
    a={}
    for i in range(len(data[0])-1):
        a[i]=[]
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            a[j].append(data[i][j])
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            data[i][j]=(data[i][j]-min(a[j]))/max(1, (max(a[j])-min(a[j])))
    return data


class NN:

    def __init__(self,data,output_dim,initialization='zero'):
        '''
        data = total dataset data with last column as target column required for binary classification
        initialization = type of initialization for weights from input layer and bias present at input layer
        columnlabel = index of target column
        layers = value of layers stored in dictionary format after applying activation_function
        shape_layers = number of neurons in ith layer
        ct_layers = total number of layers including input and output layers
        ct_weights = total number of weights vectors
        activation_functions = activation_function for each layer
        biases = bias vector for each layer
        unactivated_layers = value of layers vectors before activation
        layer_initialization = initializing each layer
        '''
        self.data=data
        self.columnlabel=len(self.data[0])-1
        self.input_layer_dim=self.columnlabel
        self.output_layer_dim = output_dim
        random.shuffle(self.data)
        self.training_data=self.data[:int(len(self.data)*0.8)]
        self.testing_data=self.data[int(len(self.data)*0.8):]
        self.layers={}
        self.shape_layers={}
        self.ct_layers=2
        self.ct_weights=1
        self.layers[1]=np.zeros([self.input_layer_dim,1])
        self.layers[2]=np.zeros([self.output_layer_dim,1])
        self.activation_functions={}
        self.activation_functions[2]='sigmoid'
        self.weights={}
        self.biases={}
        self.unactivated_layers={}
        self.layer_initialization = {}
        self.layer_initialization[1] = initialization
        self.layer_initialization[2] = initialization
        '''
        initialization of weights and biases according to the user
        '''
        if initialization=='random':
            self.weights[1]=np.random.rand(self.output_layer_dim,self.input_layer_dim)
            self.biases[1]=np.random.rand(self.output_layer_dim,1)
        elif initialization=='normal':
            self.weights[1]=np.random.randn(self.output_layer_dim,self.input_layer_dim)
            self.biases[1]=np.random.randn(self.output_layer_dim,1)
        elif initialization=='zero':
            self.weights[1]=np.zeros([self.output_layer_dim,self.input_layer_dim])
            self.biases[1]=np.zeros([self.output_layer_dim,1])
        self.shape_layers[1]=self.input_layer_dim
        self.shape_layers[2]=self.output_layer_dim

    def add_layer(self,neurons,activation_function,initialization='zero'):
        '''
        Adding layer to the neural network with given number of neurons ,
        activation_function and initialization
        '''
        self.layers[self.ct_layers+1]=self.layers[self.ct_layers]
        self.layers[self.ct_layers]=np.zeros([neurons,1])
        self.activation_functions[self.ct_layers+1]=self.activation_functions[self.ct_layers]
        self.activation_functions[self.ct_layers]=activation_function
        self.shape_layers[self.ct_layers+1]=self.shape_layers[self.ct_layers]
        self.shape_layers[self.ct_layers]=neurons
        self.layer_initialization[self.ct_layers+1]=self.layer_initialization[self.ct_layers]
        self.layer_initialization[self.ct_layers] = initialization
        self.ct_weights+=1
        self.ct_layers+=1
        for i in range(1,self.ct_weights+1):
            if self.layer_initialization[i]=='random':
                self.weights[i]=np.random.rand(self.shape_layers[i+1],self.shape_layers[i])
            elif self.layer_initialization[i]=='normal':
                self.weights[i]=np.random.randn(self.shape_layers[i+1],self.shape_layers[i])
            elif self.layer_initialization[i]=='zero':
                self.weights[i]=np.zeros([self.shape_layers[i+1],self.shape_layers[i]])
        for i in range(1,self.ct_layers):
            if self.layer_initialization[i]=='random':
                self.biases[i]=np.random.rand(self.shape_layers[i+1],1)
            elif self.layer_initialization[i]=='normal':
                self.biases[i]=np.random.randn(self.shape_layers[i+1],1)
            elif self.layer_initialization[i]=='zero':
                self.biases[i]=np.zeros([self.shape_layers[i+1],1])



    def sigmoid(self,x):
        '''
        Sigmoid Activation Function which returns 1/(1+e^-x)
        '''
        z = 1/(1 + np.exp(-x))
        return z

    def d_sigmoid(self,x):
        '''
        Differentiation of Sigmoid w.r.t x = Sigmoid(x)*(1-Sigmoid(x))
        '''
        #print(x)
        return (self.sigmoid(x))*(1-self.sigmoid(x))

    def identity(self,x):
        '''
        Indentity Activation Function
        '''
        z=x
        return z

    def d_indentity(self,x):
        '''
        Differentiation of identity w.r.t x = 1
        '''
        return np.ones((len(x),len(x[0])))

    def ReLU(self,x):
        '''
        ReLU Activation Function : (Rectified Linear Unit)
        Return max(0,a[i])
        '''
        z=[]
        l1=x.shape[0]
        l2=x.shape[1]
        for i in range(l1):
            a=[]
            for j in range(l2):
                if x[i][j]<=0:
                    a.append(0)
                else:
                    a.append(x[i][j])
            a=np.array(a)
            z.append(a)
        z=np.array(z)
        return z

    def d_ReLU(self,x):
        '''
        Differentiation of ReLU
        '''
        z=[]
        l1=x.shape[0]
        l2=x.shape[1]
        for i in range(l1):
            a=[]
            for j in range(l2):
                if x[i][j]<=0:
                    a.append(0)
                else:
                    a.append(1)
            a=np.array(a)
            z.append(a)
        z=np.array(z)
        return z

    def feedforward(self,input,expected_output):
        '''
        For All layers (except Input Layer):
            unactivated_layers[i] = weights[i-1].layers[i-1] + biases[i-1]
            layers[i] = Activation_Function[i](unactivated_layers[i])
        Predicted output = value of last layer
        '''
        for i in self.layers:
            if i==1:
                self.layers[i]=np.array(input)
                self.layers[i].reshape([self.shape_layers[i],1])
                self.unactivated_layers[i]=self.layers[i]
            else:
                if self.activation_functions[i]=='sigmoid':
                    self.layers[i]=self.weights[i-1].dot(self.layers[i-1])
                    self.layers[i]+=self.biases[i-1]
                    self.unactivated_layers[i]=self.layers[i]
                    self.layers[i]=self.sigmoid(self.layers[i])
                elif self.activation_functions[i]=='identity':
                    self.layers[i]=self.weights[i-1].dot(self.layers[i-1])
                    self.layers[i]+=self.biases[i-1]
                    self.unactivated_layers[i]=self.layers[i]
                    self.layers[i]=self.identity(self.layers[i])
                elif self.activation_functions[i]=='ReLU':
                    self.layers[i]=self.weights[i-1].dot(self.layers[i-1])
                    self.layers[i]+=self.biases[i-1]
                    self.unactivated_layers[i]=self.layers[i]
                    self.layers[i]=self.ReLU(self.layers[i])

        predicted=self.layers[self.ct_layers]
        return predicted,expected_output

    def backprop(self,predicted,expected_output):
        '''
        Backpropagation of all layers to get derivative of weights and biases
        dw = dC/dZ * dZ/dw
        db = dC/dZ * dZ/db
        '''
        deriv_w={}
        deriv_b={}
        loss=0
        for j in range(self.output_layer_dim):
            if expected_output[j][0] == 1:
                loss += -np.log(predicted[j][0])                # -expected*log(predicted)
        loss = np.array([loss])
        loss = np.array([loss])
        for i in range(self.ct_weights,0,-1):
            if i==self.ct_weights:
                if self.activation_functions[i+1]=='sigmoid':
                    deriv_w[i]=self.layers[i].dot(loss.dot(self.d_sigmoid(self.unactivated_layers[i+1]).transpose()))
                    deriv_b[i]=loss.dot(self.d_sigmoid(self.unactivated_layers[i+1]).transpose())
                elif self.activation_functions[i+1]=='identity':
                    deriv_w[i]=self.layers[i].dot(loss.dot(self.d_indentity(self.unactivated_layers[i+1]).transpose()))
                    deriv_b[i]=loss.dot(self.d_indentity(self.unactivated_layers[i+1]).transpose())
                elif self.activation_functions[i+1]=='ReLU':
                    deriv_w[i]=self.layers[i].dot(loss.dot(self.d_ReLU(self.unactivated_layers[i+1]).transpose()))
                    deriv_b[i]=loss.dot(self.d_ReLU(self.unactivated_layers[i+1]).transpose())
                deriv_w[i]=deriv_w[i].transpose()
                deriv_b[i]=deriv_b[i].transpose()
            else:
                if self.activation_functions[i+1]=='sigmoid':
                    last=self.d_sigmoid(self.unactivated_layers[i+1])
                    mid=deriv_w[i+1].transpose().dot(self.weights[i+1])
                    term=mid.dot(last)
                    term=term.transpose()
                    deriv_w[i]=self.layers[i].dot(term)
                    deriv_b[i]=term
                    deriv_w[i]=deriv_w[i].transpose()
                    deriv_b[i]=deriv_b[i].transpose()
                elif self.activation_functions[i+1]=='identity':
                    last=self.d_indentity(self.unactivated_layers[i+1])
                    mid=deriv_w[i+1].transpose().dot(self.weights[i+1])
                    term=mid.dot(last)
                    term=term.transpose()
                    deriv_b[i]=term
                    deriv_w[i]=self.layers[i].dot(term)
                    deriv_w[i]=deriv_w[i].transpose()
                    deriv_b[i]=deriv_b[i].transpose()
                elif self.activation_functions[i+1]=='ReLU':
                    last=self.d_ReLU(self.unactivated_layers[i+1])
                    mid=deriv_w[i+1].transpose().dot(self.weights[i+1])
                    term=mid.dot(last)
                    term=term.transpose()
                    deriv_b[i]=term
                    deriv_w[i]=self.layers[i].dot(term)
                    deriv_w[i]=deriv_w[i].transpose()
                    deriv_b[i]=deriv_b[i].transpose()
        return deriv_w,deriv_b

    def train_model(self,epochs=5000,batch_size=64,sample_interval=100,learning_rate=0.02):

        ep_x=[]
        lo_y=[]
        ac_y=[]
        for epoch in range(1,epochs+1):
            train_data=[]
            loss=0.
            ct=0.
            tc=0.
            for i in range(batch_size):
                idx=random.randrange(0,len(self.training_data))
                train_data.append(self.training_data[idx])
            for i in train_data:
                input=i[:len(i)-1]
                input=np.array(input)
                input=input.reshape([len(input),1])
                expected_output=i[len(i)-1]
                predicted,expected_output=self.feedforward(input,expected_output)
                # print(predicted)
                # print(expected_output)
                for j in range(self.output_layer_dim):
                    if expected_output[j][0] == 1:
                        loss += -np.log(predicted[j][0])                # -expected*log(predicted)

                tc+=1

                deriv_w,deriv_b=self.backprop(predicted,expected_output)
                '''
                Updating weights and biases according to derivative
                '''
                for i in self.weights:
                    self.weights[i]+=learning_rate*deriv_w[i]
                    self.biases[i]+=learning_rate*deriv_b[i]

                predicted_digit = -1
                prob_pred_digit = -1
                expected_digit = -1
                for j in range(self.output_layer_dim):
                    if predicted[j][0] > prob_pred_digit:
                        prob_pred_digit = predicted[j][0]
                        predicted_digit = j
                    if expected_output[j][0] == 1:
                        expected_digit = j

                if predicted_digit == expected_digit:
                    ct += 1                   

            if epoch%sample_interval==0:
                ac_y.append((ct/tc)*100)
                ep_x.append(epoch)
                lo_y.append(loss)
                print("Epoch = ",epoch,end='\t')
                print("Loss = ",loss)
                #print("Accuracy = ",(ct/tc)*100)
        self.plot(ep_x,lo_y,"loss")
        self.plot(ep_x,ac_y,"accuracy")

    def test_model(self):
        '''
        Testing the trained model on test data
        '''
        ct = 0
        tc = 0
        for i in self.testing_data:
            input=i[:len(i)-1]
            input=np.array(input)
            input=input.reshape([len(input),1])
            expected_output=i[len(i)-1]
            predicted,expected_output=self.feedforward(input,expected_output)
            predicted_digit = -1
            prob_pred_digit = -1
            expected_digit = -1
            for j in range(self.output_layer_dim):
                if predicted[j][0] > prob_pred_digit:
                    prob_pred_digit = predicted[j][0]
                    predicted_digit = j
                if expected_output[j][0] == 1:
                    expected_digit = j
            if predicted_digit == expected_digit:
                ct += 1
            tc += 1
        self.ct=ct
        self.tc=tc

    def evaluate(self):
        self.accuracy=self.ct/self.tc
        print("Accuracy = ",self.accuracy*100)

    def plot(self,x,y,title):
        '''
        Plotting the required graphs
        '''
        plt.plot(x,y)
        plt.xlabel("Epochs")
        if title=='accuracy':
            plt.ylabel("Accuracy")
            plt.title("Accuracy vs Epoch")
            plt.savefig("Accuracy_vs_epoch.png")
        else:
            plt.ylabel("Loss")
            plt.title("Loss vs Epoch")
            plt.savefig("Loss_vs_epoch.png")

        plt.show()

if __name__=='__main__':
    data=get_data()
    #print(data[0:5])
    model=NN(data,10,initialization='zero')
    model.add_layer(48,'sigmoid')
    model.add_layer(40,'sigmoid')
    model.add_layer(32,'sigmoid')
    model.add_layer(24,'sigmoid')
    model.add_layer(16,'sigmoid')
    model.train_model()
    model.test_model()
    model.evaluate()
