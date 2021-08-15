# importing required modules
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt 


# path for test file and train file
path_test="ds1_test.csv"
path_train="ds1_train.csv"

# reading files using pandas
test_file=pd.read_csv(path_test)
train_file=pd.read_csv(path_train)


class LogisticRegression:
    def __init__(self, learning_rate, n_iters):
        self.lr = learning_rate     # value of the hyper-parameter learning rate.
        self.n_iters = n_iters      # number of iterations for the training set.
        self.weights = None         
        self.bias = None
        self.j_theta=None           # array for storing values of loss function
        self.j_iter=None            # array storing the coresponding number of iterations for particular j_theta value
        
        
    # training our model     
    def fit(self, X, y):
        n_samples, n_features = X.shape    # finding number of samples in X and number of features in X 

        # init parameters declared above 
        self.weights = np.zeros(n_features)    # intializing weights to zero initially using numpy 
        self.bias = 0                            # intializing bias as zero intially using numpy 
        self.j_theta=np.zeros(self.n_iters)      # intializing j_theta as zero intially using numpy 
        self.j_iter=np.zeros(self.n_iters)      # intializing j_iter as zero intially using numpy 
        
        
        #  optimizing weights using gradient descent algorithm
        for i in range(self.n_iters):
           
            linear_model = np.dot(X, self.weights) + self.bias           # approximate y with linear combination of weights and x, plus bias
            y_predicted = self._sigmoid(linear_model)                    # apply sigmoid function
            
            
            #  calculating loss function 
            temp=0.0
            for j in range(n_samples):
                temp+=(y[j]*math.log(y_predicted[j])+(1-y[j])*math.log(1-y_predicted[j]))
            temp=(-1/n_samples)*temp
            
            self.j_theta[i]=temp        # storing value for loss function foe the ith iteration 
            self.j_iter[i]=i            # ith iteration
            
            
            # computing  gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # updating  parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        
        # plotting the loss function vs iterations graph for training set
        plt.scatter(self.j_iter,self.j_theta,label="value",color='k',s=1)
        plt.xlabel('number of iterations')
        plt.ylabel('average loss function')
        plt.title('scatter plot')
        plt.legend()
        plt.show()
        
        
    # predicting using test data after the training  
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_final = [1 if i >= 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_final)
    
    # finding sigmoid function
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def main():
  
  
    def accuracy(y_true, y_pred):    # calculating accuracy by matching predictions with test data
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    
     # coverting read data to numpy array
    X_train=train_file[['x_1','x_2']].to_numpy()       
    X_test=test_file[['x_1','x_2']].to_numpy()
     # coverting read data to numpy array
    y_train=train_file['y'].to_numpy()
    y_test=test_file['y'].to_numpy()
    
    # initailizing class as regressor 
    regressor = LogisticRegression(learning_rate=0.0005, n_iters=1000)
    # calling function to train the model 
    regressor.fit(X_train, y_train)
    # calling predict function
    predictions = regressor.predict(X_test)
    
    # printing accuracy
    print("accuracy:", accuracy(y_test, predictions))
    
    
main()