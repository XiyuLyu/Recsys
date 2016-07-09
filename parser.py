import json 
import pprint
import numpy as np
import pickle 
import os
from collections import defaultdict 

root = '/Users/cici/Desktop/DataAnalysis/'

def saveItem(item, fname):
    ''' 
        fname is the absolute path name 
    '''
    if not os.path.exists(fname):
        with open(fname, 'wb') as fp:
            pickle.dump(item, fp)


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

def getDidsByUid(fname):
    fp = open(fname,'r')
    didbyuid = defaultdict(list)
    for i, line in enumerate(fp):
        userRecords = json.loads(line)
        udoc = []
        for item in userRecords['body']['person']['preferences']:
            doc_id = item['documentId']
            if check(os.path.join(root, 'Data', 'crawls', doc_id)):
                udoc.append(doc_id)
        didbyuid[i] = udoc 

    return didbyuid

def check(doc_id):
    return True if os.path.exists(doc_id) else False 

def getDocIds(fname):
    fp = open(fname,'r')
    allDocId = []
    didMap = {}
    inverseDidMap = {}
    for line in fp.readlines():
        userRecords = json.loads(line)
        for item in userRecords['body']['person']['preferences']:
            doc_id = item['documentId']
            if check(os.path.join(root, 'Data', 'crawls', doc_id)):
                allDocId.append(doc_id)
    docId = set(allDocId)
    print len(list(docId))
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
    # print ibm.shape
    # pp = pprint.PrettyPrinter(indent = 1)
    for line in fp:
        userRecords = json.loads(line)
        # pp.pprint(userRecords)
        user_id = userRecords['id']
        for item in userRecords['body']['person']['preferences']:
            if check(os.path.join(root, 'Data', 'crawls', item['documentId'])):
                doc_id = item['documentId']
                rating = item['rating']
                i = inverseIdMap[user_id]
                j = inverseDidMap[doc_id]
                ibm[i,j] = rating

    print ibm.shape 
    if not os.path.exists('../matrix/utility.pkl'):
        fp = open('../matrix/utility.pkl' , 'wb')
        pickle.dump(ibm, fp)
    return ibm 


def getCandidateData(qname):
    qp = open(qname, 'r')
    uList = []
    cList = []
    rList = []
    for line in qp:
        items = line.strip().split('\t')
        if check(os.path.join(root, 'Data', 'crawls', items[2])):
            uList.append(int(items[0]))
            cList.append(items[2])
            rList.append(int(items[3]))
    return uList, cList, rList 

def getHistoryCandidates(fname):
    fp = open(fname, 'r')
    canMap = defaultdict(list)
    for line in fp :
        userRecords = json.loads(line)
        cand = userRecords['candidates']
        # filter out none exist files 
        cand = [ x for x in cand if check(os.path.join(root, 'Data', 'crawls', x))]
        user_id = userRecords['id']
        canMap[user_id] = cand 
    return canMap


def candidateMatrix(fname, qname):
    '''
        cm : the ground truth of user preferences over candiates 
    '''

    # fp = open(fname, 'r')
    idMap, inverseIdMap = getUserIds(fname)
    
    uList, cList, rList = getCandidateData(qname)
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

    print cm.shape

    if not os.path.exists('../matrix/canMatrix.pkl'):
        with open('../matrix/canMatrix.pkl','wb') as fp:
            pickle.dump(cm, fp)
            pickle.dump(cMap, fp)
            pickle.dump(inverseCMap, fp)
    return cm, cMap, inverseCMap
   


