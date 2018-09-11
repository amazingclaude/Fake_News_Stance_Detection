#======================================================
#===================mutual_information.py==============
#======================================================

from __future__ import division
from math import log
from collections import Counter # Counter() is a dict for counting
from collections import defaultdict
from numpy import mean
from util import *
import random
from clean import clean_delete_stopwords
from collections import Counter

base_dir='./'
#base_dir='split-data'
file_train_instances = "train_stances.csv"
file_train_bodies = "train_bodies.csv"
file_test_instances = "test_stances_unlabeled.csv"
file_test_bodies = "test_bodies.csv"
file_predictions = 'predictions_test.csv'
train = FNCData(base_dir,file_train_instances, file_train_bodies)
test_set=FNCData(base_dir,file_test_instances,file_test_bodies)

heads = []
heads_track = {}
bodies = []
bodies_track = {}
body_ids = []
#Identify unique heads and bodies
for instance in train.instances:
    head = instance['Headline']
    body_id = instance['Body ID']
    if head not in heads_track:
        heads.append(head)
        heads_track[head] = 1
    if body_id not in bodies_track:
        bodies.append(train.bodies[body_id])
        bodies_track[body_id] = 1
        body_ids.append(body_id)  

for instance in test_set.instances:
    head = instance['Headline']
    body_id = instance['Body ID']
    if head not in heads_track:
        heads.append(head)
        heads_track[head] = 1
    if body_id not in bodies_track:
        bodies.append(test_set.bodies[body_id])
        bodies_track[body_id] = 1
        body_ids.append(body_id)  
        
document=heads+bodies
 

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
    

def mutual_information(document):
    target_words = {"fake",'hoax','reportedly','claim'} #this is a set, not a dictionary
    other_words_in_the_item=set()
    corpus=set()
    N=0
    o_counts = Counter(); # Occurrence counts
    co_counts = defaultdict(Counter); 
    new_dic = defaultdict(dict)
    for item in document:
        N += 1
        words = clean_delete_stopwords(item) #This generates a words vector : ['xx','xx']
        words = set(words) #Transform the words to a set: {'xx','xx'}, set also delete duplicated words
        other_words_in_the_item= words - target_words #tranform the vector to dict.
        for word in words:
            o_counts[word] += 1 # Store occurence counts for all words
            if word not in corpus:
                corpus.add(word)               
            # but only get co-occurrence counts for target/sentiment word pairs
            if word in target_words:
                for other_word in other_words_in_the_item:
                    co_counts[word][other_word] += 1 # Store co-occurence counts
    
    other_words_in_the_corpus = corpus - target_words
    
    for target_word in target_words: 
        for test_word in other_words_in_the_corpus:   
            
            N_test_1=o_counts[test_word]
            N_test_0=N-o_counts[test_word]
            N_target_1=o_counts[target_word]
            N_target_0=N-o_counts[target_word]
            
            #N(i,j),i stands for test word, j stands for target word
            N11=co_counts[target_word][test_word]
            N10=N_test_1-N11
            N01=N_target_1-N11
            N00=N-N_target_1-N10 #Equivalent to N-N11-N10-N01
            
            #print(N_test_1,N_test_0,N_target_1,N_target_0)
            #print(N11,N_test_1,N_target_1,N)
            if test_word in co_counts[target_word]:
                mi=MI(N11,N_test_1,N_target_1,N)+MI(N10,N_test_1,N_target_0,N)+MI(N01,N_test_0,N_target_1,N)+MI(N00,N_test_0,N_target_0,N)
                new_dic[target_word][test_word]=mi
    
    for item in new_dic.items():
        d=Counter(item[1])
        print(item[0],':',d.most_common(5))
        print('=============')
    
   
mutual_information(document)
    