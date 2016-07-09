import numpy as np 
import pickle
from parser import * 
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer 

root = '/Users/cici/Desktop/DataAnalysis/'
def loadData():
    fp = open('../matrix/utility.pkl', 'r')
    qp = open('../matrix/canMatrix.pkl','r')
    hdata = pickle.load(fp)
    rdata = pickle.load(qp)
    cMap = pickle.load(qp)
    inverseCMap = pickle.load(qp)
    return hdata, rdata, cMap, inverseCMap  

def binarize(data):
    '''
        rating is -1 ~ 4
        -1, 0 , 1 ->  0
        2, 3 , 4 ->   1
    '''
    hdata[hdata < 2] = 0
    hdata[hdata >= 2] = 1
    return hdata 

def historyTFIDF(fname, qname):
    didMap , dummpy = getDocIds(fname)
    tfidf = TfidfVectorizer(max_features = 100)
    history = []
    candidate = []
    dummpy, cList, dummpy = getCandidateData(qname)
    cList = list(set(cList))
    count = 0
    for v in didMap.values():
        fn = os.path.join(root, 'Data', 'crawls', v)
        if os.path.exists(fn):
            contents = open(fn, 'r').read()
            history.append(contents)
    #print history 
    train = tfidf.fit_transform(history)
    if not os.path.exists('../matrix/train_100.pkl'):
        with open('../matrix/train_100.pkl', 'rb') as fp:
            pickle.dump(train, fp)


    for v in cList : 
        fn = os.path.join(root, 'Data', 'crawls', v)
        if os.path.exists(fn):
            contents = open(fn, 'r').read()
            candidate.append(contents)

    
    test = tfidf.transform(candidate)
    if not os.path.exists('../matrix/test_100.pkl'):
        with open('../matrix/test_100.pkl', 'rb') as fp:
            pickle.dump(test, fp)

    print train.shape, test.shape



