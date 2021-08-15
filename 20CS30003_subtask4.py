# importing required modules
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt 


# path for test file and train file
path_test="ds2_test.csv"
path_train="ds2_train.csv"

# reading files using pandas
test_file=pd.read_csv(path_test)
train_file=pd.read_csv(path_train)


class GaussianDiscriminantAnalysis:
    def __init__(self):
        pass
        # calculate phi
    def calculate_phi(self,Y):
        m = Y.shape
        return np.sum(Y)/m
  


    
# calculate mue1
    def calculate_mu1(self,X,Y):
        m=Y.shape
        
        y_pos=np.sum(Y)
        
        conditional_sum_x=np.zeros([1,2])
        for i in range(m[0]):
            xi=X[1,:]
            yi=Y[i]
            conditional_sum_x=conditional_sum_x+(xi if yi==1 else np.zeros([1,2]))
            
        return (1/m[0])*conditional_sum_x/y_pos
        
        
        # calculate mue0
    def calculate_mu0(self,X,Y):
        m=Y.shape
        y_neg=m-np.sum(Y)
        
        conditional_sum_x=np.zeros([1,2])
        for i in range(m[0]):
            xi=X[1,:]
            yi=Y[i]
            conditional_sum_x=conditional_sum_x + (xi if yi==0 else np.zeros([1,2]))
            
        return (1/m[0])*conditional_sum_x/y_neg
        
   # calculate sigma
    def calculate_sigma(self,X,Y,mu0,mu1):
        m = Y.shape
        sqr_sum=np.zeros([2,2])
        for i in range(m[0]):
            xi=X[i,:]
            yi=Y[i]
            sqr= (np.dot((xi-mu1),(xi-mu1).T) if yi==1 else np.dot((xi-mu0),(xi-mu0).T))
            sqr_sum=sqr_sum+sqr
        return sqr_sum/m[0]
 
 
 # calculate probability
    def prob(self,X,Y,mu0,mu1,sigma,phi):
        m=Y.shape
        theta=np.dot(np.linalg.inv(sigma),np.sum(mu1-mu0))
        
        theta_nod=(0.5)*np.dot(mu0,np.linalg.inv(sigma),mu0)+(-0.5)*np.dot(mu1,np.linalg.inv(sigma),mu1)-math.log((1-phi)/(phi))
       
        proba=np.zeros([m[0],1])
        for i in range(m[0]):
            temp=1/(1+math.exp(-np.dot(theta.T,X[i,:])-theta_nod))
            proba[i]=1 if temp>=0.5 else 0
        return proba
    
        



def main():


#calculate accuracy
    def accuracy(y_true, y_pred):    # calculating accuracy by matching predictions with test data
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
        
        
     # coverting read data to numpy array
    X_train=train_file[['x_1','x_2']].to_numpy()       
    X_test=test_file[['x_1','x_2']].to_numpy()
     # coverting read data to numpy array
    y_train=train_file['y'].to_numpy()
    y_test=test_file['y'].to_numpy()
    
    #initialize gda
    recur=GaussianDiscriminantAnalysis()
    # phi
    phi=recur.calculate_phi(y_train)
    # mu1
    mu1=recur.calculate_mu1(X_train,y_train)
    mu0=recur.calculate_mu0(X_train,y_train)# mu0
    
    sigma=recur.calculate_sigma(X_train,y_train,mu0,mu1)# sigma
    
    
    proba=recur.prob(X_train,y_train,mu0,mu1,sigma,phi)# probabilitty
    
    accur=accuracy(proba,y_train)# accuracy 
    
    print("accuracy",accur)
    
main()