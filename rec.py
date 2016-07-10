import numpy as np 
import pickle
from parser import * 
from sklearn.feature_extraction.text import  TfidfVectorizer

root = '/Users/cici/Desktop/DataAnalysis/'

def loadUtility():
    fp = open('../matrix/utility.pkl', 'r')
    hdata = pickle.load(fp)
    return hdata 

def loadCanMatrix():
    qp = open('../matrix/canMatrix.pkl','r')    
    cdata = pickle.load(qp)
    cMap = pickle.load(qp)
    inverseCMap = pickle.load(qp)
    return cdata, cMap, inverseCMap  

def normalize(hdata):
    hdata = hdata/5.0 + 0.2
    return hdata 

def historyTFIDF(fname, qname):
    didMap , dummy = getDocIds(fname)
    tfidf = TfidfVectorizer(max_features = 100)
    history = []
    candidate = []
    dummy, cList, dummy = getCandidateData(qname)
    cList = list(set(cList))
    # count = 0
    for i, v in enumerate(didMap.values()):
        '''
            deal with histories
        ''' 
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
        '''
            deal with candidates
        '''         

        fn = os.path.join(root, 'Data', 'crawls', v)
        # if i % 100 == 0 :
        #     print i 
        if os.path.exists(fn):
            contents = open(fn, 'r').read()
            candidate.append(contents)

    
    test = tfidf.transform(candidate)
    if not os.path.exists('../matrix/test_100.pkl'):
        with open('../matrix/test_100.pkl', 'wb') as fp:
            pickle.dump(test, fp)

    print test.shape

def adjustedCosine(a, b):
    ua = np.mean(a)
    ub = np.mean(b)
    denominator = np.sqrt(np.dot((a - ua), (a-ua) ) * np.dot((b-ub), (b-ub)) )
    return np.dot((a-ua), (b-ub))/ denominator

def hCSimilarity(train_fn, test_fn, save_fn):
    '''
        calculate History and Candidate matrix similarity
        n x 100 , m x 100 -> n x m 
        #@TODO :
            there are a lot of ways to calcaute similarity
    '''

    fp = open(train_fn, 'r')
    train = pickle.load(fp).toarray()
    fp = open(test_fn , 'r')
    test  = pickle.load(fp).toarray()
    result = np.zeros((len(train), len(test)))

    for i in range(len(train)):
        for j in range(len(test)):
            result[i][j] = adjustedCosine(train[i,:], test[j,:])
            
    saveItem(result, save_fn)

    # if not os.path.exists('../matrix/item_item_similarity.pkl'):
    #     with open('../matrix/item_item_similarity.pkl', 'wb') as fp:
    #         pickle.dump(result, fp)

def evaluateRankMean(item):
    k = 5 
    return 1 if np.mean(np.sort(item)[::-1][:k]) > 0.5 else 0 

def evaluateByMean(item):
    return 1 if np.mean(item) > 0.5 else 0

def evaluateWeightedMean(item, rating):
    # k = 5 
    # topkIndex = np.argsort(item)[::-1][:k] 
    # item = item[topkIndex]
    # rating = rating[topkIndex]
    rating = normalize(rating)
    #print np.dot(item, rating), np.sum(item)
    return 1 if (np.dot(item, rating) + 0.01)/(np.sum(item) + 0.01) > 0.5 else 0 


def rateCan(fname, similarity_fn):
    idMap, inverseIdMap = getUserIds(fname)
    didMap, inverseDidMap = getDocIds(fname)
    historyCanMap = getHistoryCandidates(fname)
    didbyuid = getDidsByUid(fname)
    with open('../matrix/utility.pkl', 'rb') as fp:
        utility = pickle.load(fp)

    with open('../matrix/canMatrix.pkl','rb') as fp:
        cm = pickle.load(fp)
        cMap = pickle.load(fp) 
        inverseCMap  = pickle.load(fp)

    with open(similarity_fn, 'rb') as fp:
        item_item_similarity = pickle.load(fp)

    prediction = np.zeros(cm.shape)

    tp = 0
    fp = 0

    for uk, uv in idMap.items():
        candidates = historyCanMap[uv]
        preferences = didbyuid[uk]
        preferences_ids = [ inverseDidMap[k] for k in preferences]
        rating = utility[uk, preferences_ids]
        for candidate in candidates:
            can_id = inverseCMap[candidate]
            tmp = item_item_similarity[preferences_ids , can_id]
            
            #predict_uid_cid= 1 if np.mean(tmp)  > 0.5 else 0
            # predict_uid_cid = evaluateRankMean(tmp)
            predict_uid_cid = evaluateWeightedMean(tmp, rating)
            if predict_uid_cid == 1 and cm[uk, can_id] == 0:
                fp = fp + 1
            elif predict_uid_cid == 1 and cm[uk, can_id] == 1:
                tp = tp + 1
            prediction[uk, can_id] = predict_uid_cid

    #from sklearn.metrics import precision_score
    #print 'precision is {}'.format(precision_score(np.reshape(cm,1, np.prod(cm.shape)), np.reshape(prediction, 1, np.prod(cm.shape))))
    print 'precision is {}'.format(tp * 1.0/(tp + fp))

    #print 'error rate : {}'.format(error * 1.0/211.0/30.0) 
 
    saveItem(prediction, os.path.join(root, 'matrix', 'prediction.pkl'))

