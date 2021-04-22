import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

def get_data(filename):
    '''
    Reads the filename given as argument and converts csv to data array and return normalized data
    '''
    import csv
    data=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count=0
        for row in csv_reader:
            if line_count!=0:
                a=[]
                for i in range(len(row)-1):
                    a.append(float(row[i]))
                a.append(int(row[len(row)-1]))
                data.append(a)
            line_count+=1
    return normalize_data(data)

def normalize_data(data):
    '''
    Normalizes the dataset
    X[i] = X[i] - min(ith column) / (max(ith column) - min(ith column))
    '''
    a={}
    for i in range(len(data[0])):
        a[i]=[]
    for i in range(len(data)):
        for j in range(len(data[i])):
            a[j].append(data[i][j])
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=(data[i][j]-min(a[j]))/(max(a[j])-min(a[j]))
    return data


class NN:
    '''
    Class Implementing Neural Networks from scratch
    with all the functions such as add_layer , train_model ...
    It takes data as input , splits the data into test-train ,
    creates layers as specified with activation functions and number of neurons
    per layer.
    '''
    def __init__(self,data,initialization='random'):
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
        '''
        self.data=data
        self.columnlabel=len(self.data[0])-1
        self.input_layer_dim=self.columnlabel
        random.shuffle(self.data)
        self.training_data=self.data[:int(len(self.data)*0.8)]
        self.testing_data=self.data[int(len(self.data)*0.8):]
        self.layers={}
        self.shape_layers={}
        self.ct_layers=2
        self.ct_weights=1
        self.layers[1]=np.zeros([self.input_layer_dim,1])
        self.layers[2]=np.zeros([1,1])
        self.activation_functions={}
        self.activation_functions[2]='sigmoid'
        self.weights={}
        self.biases={}
        self.unactivated_layers={}
        '''
        initialization of weights and biases according to the user
        '''
        if initialization=='random':
            self.weights[1]=np.random.rand(1,self.input_layer_dim)
            self.biases[1]=np.random.rand(1,1)
        elif initialization=='normal':
            self.weights[1]=np.random.randn(1,self.input_layer_dim)
            self.biases[1]=np.random.randn(1,1)
        elif initialization=='zero':
            self.weights[1]=np.zeros([1,self.input_layer_dim])
            self.biases[1]=np.zeros([1,1])
        self.shape_layers[1]=self.input_layer_dim
        self.shape_layers[2]=1

    def add_layer(self,neurons,activation_function,initialization='random'):
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
        self.ct_weights+=1
        self.ct_layers+=1
        for i in range(1,self.ct_weights+1):
            if initialization=='random':
                self.weights[i]=np.random.rand(self.shape_layers[i+1],self.shape_layers[i])
            elif initialization=='normal':
                self.weights[i]=np.random.randn(self.shape_layers[i+1],self.shape_layers[i])
            elif initialization=='zero':
                self.weights[i]=np.zeros([self.shape_layers[i+1],self.shape_layers[i]])
        for i in range(1,self.ct_layers):
            if initialization=='random':
                self.biases[i]=np.random.rand(self.shape_layers[i+1],1)
            elif initialization=='normal':
                self.biases[i]=np.random.randn(self.shape_layers[i+1],1)
            elif initialization=='zero':
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
        return (self.sigmoid(x))*(1-self.sigmoid(x))

    def tanh(self,x):
        '''
        Tanh Activation Function
        '''
        z=np.tanh(x)
        return z

    def d_tanh(self,x):
        '''
        Differentiation of Tanh w.r.t x =(1-(Tanh(x))^2)
        '''
        return 1-(self.tanh(x))**2

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
        FeedForwarding using calculated weights to obtain layers vectors
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
                elif self.activation_functions[i]=='tanh':
                    self.layers[i]=self.weights[i-1].dot(self.layers[i-1])
                    self.layers[i]+=self.biases[i-1]
                    self.unactivated_layers[i]=self.layers[i]
                    self.layers[i]=self.tanh(self.layers[i])
                elif self.activation_functions[i]=='ReLU':
                    self.layers[i]=self.weights[i-1].dot(self.layers[i-1])
                    self.layers[i]+=self.biases[i-1]
                    self.unactivated_layers[i]=self.layers[i]
                    self.layers[i]=self.ReLU(self.layers[i])

        predicted=self.layers[self.ct_layers]
        expected_output=np.array([expected_output])
        expected_output=np.array([expected_output])
        return predicted,expected_output

    def backprop(self,predicted,expected_output):
        '''
        Backpropagation of all layers to get derivative of weights and biases
        dw = dC/dZ * dZ/dw
        db = dC/dZ * dZ/db
        '''
        deriv_w={}
        deriv_b={}
        loss=predicted-expected_output
        for i in range(self.ct_weights,0,-1):
            if i==self.ct_weights:
                if self.activation_functions[i+1]=='sigmoid':
                    deriv_w[i]=self.layers[i].dot((expected_output-predicted).dot(self.d_sigmoid(self.unactivated_layers[i+1])))
                    deriv_b[i]=(expected_output-predicted).dot(self.d_sigmoid(self.unactivated_layers[i+1]))
                elif self.activation_functions[i+1]=='tanh':
                    deriv_w[i]=self.layers[i].dot((expected_output-predicted).dot(self.d_tanh(self.unactivated_layers[i+1])))
                    deriv_b[i]=(expected_output-predicted).dot(self.d_tanh(self.unactivated_layers[i+1]))
                elif self.activation_functions[i+1]=='ReLU':
                    deriv_w[i]=self.layers[i].dot((expected_output-predicted).dot(self.d_ReLU(self.unactivated_layers[i+1])))
                    deriv_b[i]=(expected_output-predicted).dot(self.d_ReLU(self.unactivated_layers[i+1]))
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
                elif self.activation_functions[i+1]=='tanh':
                    last=self.d_tanh(self.unactivated_layers[i+1])
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

    def train_model(self,epochs=10000,batch_size=128,sample_interval=100,learning_rate=0.01):
        '''
        Trains model for given number of epochs taking batch_size rows everytime in 1 epoch.
        It adds learning_rate times derivative to weights and biases to reach local minima.
        It prints loss and accuracy at every sample_interval intervals
        '''
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
                '''
                Calculating loss using binary crossentropy for binary classification
                Loss  = - i * Log(P[i]) - (1 - i) * Log(1 - P[i])
                '''
                try:
                    if expected_output==0:
                        loss+=-np.log(1-predicted)
                    else:
                        loss+=-np.log(predicted)
                except Exception as e:
                    print(e,predicted,1-predicted)
                tc+=1

                deriv_w,deriv_b=self.backprop(predicted,expected_output)
                '''
                Updating weights and biases according to derivative
                '''
                for i in self.weights:
                    self.weights[i]+=learning_rate*deriv_w[i]
                    self.biases[i]+=learning_rate*deriv_b[i]
                '''
                Predicting the output using threshhold as 0.5
                '''
                if predicted<=0.5:
                    predicted=0
                else:
                    predicted=1
                if predicted==expected_output:
                    ct+=1

            if epoch%sample_interval==0:
                ac_y.append((ct/tc)*100)
                ep_x.append(epoch)
                lo_y.append(loss[0][0])
                print("Epoch = ",epoch,end='\t')
                print("Loss = ",loss,end='\t')
                print("Accuracy = ",(ct/tc)*100)
        self.plot(ep_x,lo_y,"loss")
        self.plot(ep_x,ac_y,"accuracy")

    def test_model(self):
        '''
        Testing the trained model on test data Calculating accuracy and f-score
        '''
        tp=0.
        fp=0.
        tn=0.
        fn=0.
        for i in self.testing_data:
            input=i[:len(i)-1]
            input=np.array(input)
            input=input.reshape([len(input),1])
            expected_output=i[len(i)-1]
            predicted,expected_output=self.feedforward(input,expected_output)
            if(predicted<=0.5):
                predicted=0
            else:
                predicted=1
            expected_output=int(expected_output[0][0])
            if expected_output==1:
                if predicted==1:
                    tp+=1
                else:
                    fn+=1
            else:
                if predicted==1:
                    fp+=1
                else:
                    tn+=1
        self.tp=tp
        self.tn=tn
        self.fp=fp
        self.fn=fn

    def evaluate(self):
        '''
        Evaluating the model using confusion matrix
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F-Score = (2 * precision * recall) / (precision + recall)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        '''
        self.precision=self.tp/(self.tp+self.fp)
        self.recall=self.tp/(self.tp+self.fn)
        self.f_score=(2*self.precision*self.recall)/(self.precision+self.recall)
        self.accuracy=(self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)
        print("Accuracy = ",self.accuracy)
        print("F-Score = ",self.f_score)

    def check_model(self):
        '''
        Just to check the model and shapes
        '''
        print(self.ct_layers)
        print(self.shape_layers)
        print(self.layers)
        print(self.activation_functions)
        print(self.ct_weights)
        print(self.weights)
        print(self.biases)

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
    filename='housepricedata.csv'
    data=get_data(filename)
    model=NN(data,initialization='normal')
    model.add_layer(6,'sigmoid')
    model.add_layer(4,'sigmoid')
    model.train_model()
    model.test_model()
    model.evaluate()
