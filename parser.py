import json 
#from settings from *

import pprint

def getUserIds(fname):
    fp = open(fname,'r')
    idMap = {}
    for i, line in enumerate(fp):
        userRecords = json.loads(line)
        user_id = userRecords['id']
        idMap[i] = user_id
    return idMap

def getDocIds(fname):
    fp = open(fname,'r')
    allDocId = []
    didMap = {}
    for line in fp:
        userRecords = json.loads(line)
        for item in userRecords['body']['person']['preferences']:
            doc_id = item['documentId']
            rating = item['rating']
            allDocId.append(doc_id)
    docId = set(allDocId)
    for i, did in enumerate(docId):
        didMap[i] = did
    return didMap



def dataParser(fname):
    '''
        input : batch_requrest.json file
        output : a 211 x 7352 matrix
    '''
    fp = open(fname, 'r')
    data = []
    pp = pprint.PrettyPrinter(indent = 1)
    for line in fp:
        userRecords = json.loads(line)
        pp.pprint(userRecords)
        user_id = userRecords['id']
        for item in userRecords['body']['person']['preferences']:
            doc_id = item['documentId']
            rating = item['rating']



        # print data
        break
    # return data 
    
    
    
    
    