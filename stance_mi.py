#======================================================
#===================stance_mi.py==============
#======================================================

from __future__ import division
from util import FNCData
from clean import clean_delete_stopwords
from collections import Counter # Counter() is a dict for counting
from collections import defaultdict
from numpy import mean
from math import log


file_train_instances = "train_stances.csv"
file_train_bodies = "train_bodies.csv"
train = FNCData('./',file_train_instances, file_train_bodies)
# Initialise


Stance=['agree','disagree','discuss','unrelated']
train_set = []
train_stances = []

head_token_track = {}
body_token_track = {}

N=0
o_counts = Counter(); # Occurrence counts
class_counts=Counter();
co_counts = defaultdict(Counter); 

corpus=set()
new_dic = defaultdict(dict)
# Process train set
for instance in train.instances:
    
    
    head = instance['Headline']
    body_id = instance['Body ID']
   
    if head not in head_token_track:
        head_token = clean_delete_stopwords(head)
        head_token_track[head] = head_token
    else:
        head_token = head_token_track[head]
    if body_id not in body_token_track:
        body_token = clean_delete_stopwords(train.bodies[body_id])
        body_token_track[body_id] = body_token
    else:
        body_token = body_token_track[body_id]
    
    class_counts[instance['Stance']]+=1
    all_tokens=head_token+body_token
    all_tokens=set(all_tokens)
    N+=1
    for word in all_tokens:
        if word not in corpus:
            corpus.add(word)
        o_counts[word]+=1
        co_counts[instance['Stance']][word]+=1
        
def MI(c_xy, c_x, c_y, N):
    # Computes PMI(x, y) where
    # c_xy is the number of times x co-occurs with y
    # c_x is the number of times x occurs.
    # c_y is the number of times y occurs.
    # N is the number of observations.
    if c_x*c_y==0 or c_xy==0:
        mi=0
    else: mi=(c_xy/N)*log(N*c_xy/(c_x*c_y), 2)
    return mi
    
for stance in Stance:
    for word in corpus:
        N_test_1=o_counts[word]
        N_test_0=N-o_counts[word]
        N_class_1=class_counts[stance]
        N_class_0=N-class_counts[stance]   
        
        #N(i,j),i stands for test word, j stands for class word
        N11=co_counts[stance][word]
        N10=N_test_1-N11
        N01=N_class_1-N11
        N00=N-N_class_1-N10 #Equivalent to N-N11-N10-N01
        
        
        if word in co_counts[stance]: #to avoid co_counts[stance][word]=0 and causes error in mi calculation_
            mi=MI(N11,N_test_1,N_class_1,N)+MI(N10,N_test_1,N_class_0,N)+MI(N01,N_test_0,N_class_1,N)+MI(N00,N_test_0,N_class_0,N)
            new_dic[stance][word]=mi
            
for item in new_dic.items():
    d=Counter(item[1])
    print(item[0],':',d.most_common(5))
    print('=============')
    


