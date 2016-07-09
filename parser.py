import json 
import pprint
import numpy as np
import pickle 
import os
from collections import defaultdict 
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
        fp = open('../matrix/utility.pkl' , 'wb')
        pickle.dump(ibm, fp)
    return ibm 


def candidateMatrix(fname, qname):
    # fp = open(fname, 'r')
    idMap, inverseIdMap = getUserIds(fname)
    # canMap = defaultdict(list)
    # for line in fp :
    #     userRecords = json.loads(line)
    #     cand = userRecords['candidates']
    #     user_id = userRecords['id']
    #     canMap[user_id] = cand 

    qp = open(qname, 'r')
    uList = []
    cList = []
    rList = []
    for line in qp:
        items = line.strip().split('\t')
        uList.append(int(items[0]))
        cList.append(items[2])
        rList.append(int(items[3]))

    # print len(dList), len(list(set(dList)))
    nuList = list(set(uList))
    ncList = list(set(cList))
    cm = np.zeros((len(nuList),len(ncList)))
    cMap = {}
    inverseCMap = {}
    for i, item in enumerate(ncList):
        cMap[i] = item
        inverseCMap[item] = i 
         
    for i, (uid, cid) in enumerate(zip(uList, cList)):
        cm[inverseIdMap[uid],inverseCMap[cid]] = rList[i]
    print cm
   


