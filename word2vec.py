#======================================================
#===================word2vec.py==============
#======================================================

from clean import clean_delete_stopwords
import gensim
import numpy as np
from scipy import spatial
import pyemd #for WMD algorithm calculation


def avg_feature_vector(sentence,model, num_features, index2word_set):
     
    words = clean_delete_stopwords(sentence)
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def word2vec_cos(title, body,model,index2word_set):
    

    s1_afv = avg_feature_vector(title,model=model, num_features=300, index2word_set=index2word_set)
    s2_afv = avg_feature_vector(body,model=model, num_features=300, index2word_set=index2word_set)
    #Here to detect weather a numpy array contains all zero (which will causes cosine similarity NaN problem, with demoninator as zero)
    if (not np.any(s1_afv)) and (not np.any(s2_afv)):
        dis = spatial.distance.cosine(s1_afv, s2_afv)
        sim=1-dis
    else:
        dis=1
        sim=0
    return dis,sim

def wmd_distance(title,body,model):
    if (title!='') and (body!=''):
        distance = model.wmdistance(title, body)
    else:
        distance=10
    return distance