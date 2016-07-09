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

historyTFIDF(fname, qname)
#testBinary()
#testLoader()
# testDidMap()
# testIdMap()
