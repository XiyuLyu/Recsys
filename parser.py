import json 
import pprint
import numpy as np
import pickle 
import os

def getUserIds(fname):
    fp = open(fname,'r')
    idMap = {}
    inverseIdMap = {}
    for i, line in enumerate(fp):
        userRecords = json.loads(line)
        user_id = userRecords['id']
        idMap[i] = user_id
        inverseIdMap[user_id] = i 
    return idMap, inverseIdMap

def getDocIds(fname):
    fp = open(fname,'r')
    allDocId = []
    didMap = {}
    inverseDidMap = {}
    for line in fp:
        userRecords = json.loads(line)
        for item in userRecords['body']['person']['preferences']:
            doc_id = item['documentId']
            allDocId.append(doc_id)
    docId = set(allDocId)
    for i, did in enumerate(docId):
        didMap[i] = did
        inverseDidMap[did] = i
    return didMap, inverseDidMap

def dataParser(fname):
    '''
        input : batch_requrest.json file
        output : a 211 x  matrix
    '''
    fp = open(fname, 'r')
    data = []
    idMap, inverseIdMap = getUserIds(fname)
    didMap, inverseDidMap = getDocIds(fname)
    ibm = np.zeros((len(idMap), len(didMap)))
    print ibm.shape
    # pp = pprint.PrettyPrinter(indent = 1)
    for line in fp:
        userRecords = json.loads(line)
        # pp.pprint(userRecords)
        user_id = userRecords['id']
        for item in userRecords['body']['person']['preferences']:
            doc_id = item['documentId']
            rating = item['rating']
            ibm[inverseIdMap[user_id],inverseDidMap[doc_id]] = rating

    if not os.path.exists('../matrix/utility.pkl'):
        with open('../matrix/utility.pkl' , 'wb') as fp :
            pickle.dump(ibm, fp)
    return ibm 
    
    
    
    
    