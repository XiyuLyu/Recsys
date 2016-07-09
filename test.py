from parser import *

root = '/Users/cici/Desktop/DataAnalysis/'
import os
fname = os.path.join(root, 'Data', 'batch_requests.json') 
def testLoader():
    
    data = dataParser(fname)
    # print data 
def testDidMap():
    print getDocIds(fname)
def testIdMap():
    print getUserIds(fname)   
testLoader()
# testDidMap()
# testIdMap()
