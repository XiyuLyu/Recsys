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
    for i, v in enumerate(didMap.values()):
        fn = os.path.join(root, 'Data', 'crawls', v)
        if os.path.exists(fn):
            contents = open(fn, 'r').read()
            history.append(contents)
    #print history 
    train = tfidf.fit_transform(history)
    if not os.path.exists('../matrix/train_100.pkl'):
        with open('../matrix/train_100.pkl', 'wb') as fp:
            pickle.dump(train, fp)

    print train.shape 

    for i, v in enumerate(cList): 
        fn = os.path.join(root, 'Data', 'crawls', v)
        if i % 100 == 0 :
            print i 
        if os.path.exists(fn):
            contents = open(fn, 'r').read()
            candidate.append(contents)

    
    test = tfidf.transform(candidate)
    if not os.path.exists('../matrix/test_100.pkl'):
        with open('../matrix/test_100.pkl', 'wb') as fp:
            pickle.dump(test, fp)

    print test.shape


def hCSimilarity(train_fn, test_fn):
    '''
        calculate History and Candidate matrix similarity
        n x 100 , m x 100 -> n x m 
    '''

    fp = open(train_fn, 'r')
    train = pickle.load(fp).toarray()
    fp = open(test_fn , 'r')
    test  = pickle.load(fp).toarray()
    result = np.zeros((len(train), len(test)))

    for i in range(len(train)):
        for j in range(len(test)):
            result[i][j] = np.dot(train[i,:], test[j,:])


    if not os.path.exists('../matrix/item_item_similarity.pkl'):
        with open('../matrix/item_item_similarity.pkl', 'wb') as fp:
            pickle.dump(result, fp)


def rateCan(fname):
    idMap, inverseIdMap = getUserIds(fname)
    didMap, inverseDidMap = getDocIds(fname)
    historyCanMap = getHistoryCandidates(fname)
    didbyuid = getDidsByUid(fname)

    with open('../matrix/canMatrix.pkl','rb') as fp:
        cm = pickle.load(fp)
        cMap = pickle.load(fp) 
        inverseCMap  = pickle.load(fp)

    with open('../matrix/item_item_similarity.pkl', 'rb') as fp:
        item_item_similarity = pickle.load(fp)

    prediction = np.zeros(cm.shape)

    error = 0
    for uk, uv in idMap.items():
        candidates = historyCanMap[uv]
        preferences = didbyuid[uk]
        preferences_ids = [ inverseDidMap[k] for k in preferences]

        for candidate in candidates:
            can_id = inverseCMap[candidate]
            tmp = item_item_similarity[preferences_ids , can_id]
            predict_uid_cid= 1 if np.mean(tmp)  > 0.5 else 0
            if predict_uid_cid != cm[uk, can_id]:
                error = error + 1
            prediction[uk, can_id] = predict_uid_cid


    print 'error rate : {}'.format(error * 1.0/211.0/30.0) 
 
    saveItem(prediction, os.path.join(root, 'matrix', 'prediction.pkl'))

