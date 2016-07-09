import numpy as np 
import pickle

def loadData():
    fp = open('../matrix/utility.pkl', 'r')
    data = pickle.load(fp) 
    return data 

def binarize(data):
    '''
        rating is -1 ~ 4
        -1, 0 , 1 ->  0
        2, 3 , 4 ->   1
    '''
    data[data < 2] = 0
    data[data >= 2] = 1
    return data 

def ItemRec(data):
    return None







