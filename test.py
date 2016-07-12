from parser import *
from rec import * 
root = '/Users/cici/Desktop/DataAnalysis/'
import os
fname = os.path.join(root, 'Data', 'batch_requests.json') 
qname = os.path.join(root, 'Data', 'qrel')
def testLoader():
    
    data = dataParser(fname)
    # print data 
def testDidMap():
    print getDocIds(fname)

def testIdMap():
    print getUserIds(fname)   

def testBinary():
    data = loadData()
    bn = binarize(data)
    print bn 

def testCan():
    candidateMatrix(fname, qname)

def testHCSimilarity():
    f1 = '../matrix/train_1000.pkl'
    f2 = '../matrix/test_1000.pkl'
    hCSimilarity(f1, f2, '../matrix/item_item_similarity.pkl')


historyMatrix(fname)
candidateMatrix(fname, qname)
historyTFIDF(fname, qname)
testHCSimilarity()
rateCan(fname, '../matrix/item_item_similarity.pkl')
rateTopK(5,fname)
