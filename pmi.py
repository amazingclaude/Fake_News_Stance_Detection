#======================================================
#===================pmi.py=============================
#======================================================

#Adapted from http://homepages.inf.ed.ac.uk/sgwater/teaching/lsa2015/labs/lab4.py
#Original credit - @Sharon Goldwater

from __future__ import division
from math import log
from collections import Counter # Counter() is a dict for counting
from collections import defaultdict
from numpy import mean
from util import *
import random
from clean import clean_delete_stopwords
base_dir='./'
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
# Identify unique heads and bodies
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

# for instance in test_set.instances:
    # head = instance['Headline']
    # body_id = instance['Body ID']
    # if head not in heads_track:
        # heads.append(head)
        # heads_track[head] = 1
    # if body_id not in bodies_track:
        # bodies.append(test_set.bodies[body_id])
        # bodies_track[body_id] = 1
        # body_ids.append(body_id)  
        
document=heads+bodies


# List of key words:
topic_words_list1 = ['fake','hoax','fraud']
# List of key words:
topic_words_list2 = ['reportedly','report','according']


sentiment_words = topic_words_list1+topic_words_list2

def PMI(c_xy, c_x, c_y, N):
    # Computes PMI(x, y) where
    # c_xy is the number of times x co-occurs with y
    # c_x is the number of times x occurs.
    # c_y is the number of times y occurs.
    # N is the number of observations.
    
    return  log(N*c_xy/(c_x*c_y), 2)
    
#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

#remove any keys from counts dictionary unless their count is above min_threshold
#if max_threshold is set, also remove anything whose count is equal to or above that threshold 
def filter_o_counts(counts, min_threshold, max_threshold=0):
    if (max_threshold > 0):
        return Counter({w : counts[w] for w in counts.keys() if counts[w] > min_threshold and counts[w] < max_threshold})
    else:
        return Counter({w : counts[w] for w in counts.keys() if counts[w] > min_threshold})

#remove any co-occ. counts if they are not above threshold 
def filter_co_counts(co_counts, threshold):
     return {w: filter_o_counts(co_counts[w], threshold) for w in co_counts.keys()}

#train is FNCData("train_stances.csv","train_bodies.csv")

#def pointwise_mutual_information(document):
corpus_without_sentiments=set()

# Define the data structures used to store the counts:
o_counts = Counter(); # Occurrence counts
co_counts = defaultdict(Counter); # Co-occurrence counts:
  #This will be indexed by target words. co_counts[target] will contain
  #a dictionary of co-occurrence counts of target with each sentiment word.

N = 0 #This will store the total number of observations (title/body)
      # You should add code to the block below so that N has the
      # correct value when the block finishes.
# Load the data:

for item in document:
    N += 1
    words = clean_delete_stopwords(item)
    for word in words:
        o_counts[word] += 1 # Store occurence counts for all words
        # but only get co-occurrence counts for target/sentiment word pairs   
        if word not in sentiment_words:
            corpus_without_sentiments.add(word)
            for word2 in words:
                if word2 in sentiment_words:
                    co_counts[word][word2] += 1 # Store co-occurence counts
                
print("Total number of documents(title or body): {}".format(N))



#filter out co-occurrences with too few counts
#co_counts = filter_co_counts(co_counts,0)


pmi_refuting_dict=defaultdict(dict)
pmi_discussion_dict=defaultdict(dict)
for target in corpus_without_sentiments:
    target_count = o_counts[target]
    topic1_PMIs = []
    topic2_PMIs = []
    # compute PMI between target and each positive word, and
    # add it to the list of fake sentimental orientation PMI values
    for keyword1 in topic_words_list1:
        if(keyword1 in co_counts[target]): # Check if the words actually co-occur
            # If so, compute PMI and append to the list
            if co_counts[target][keyword1]>3:
                topic1_PMIs.append(PMI(co_counts[target][keyword1],target_count,o_counts[keyword1],N))
        
    # same for discuss sentimental orientation words
    for keyword2 in topic_words_list2:
        if(keyword2 in co_counts[target]): 
            if co_counts[target][keyword2]>3:
                topic2_PMIs.append(PMI(co_counts[target][keyword2],target_count,o_counts[keyword2],N))
       
#uncomment the following line when topic1_PMIs and topic2_PMIs are no longer empty.
    #print("{} {:.2f} (keyword1), {:.2f} (keyword2)".format((target+":").ljust(12), mean(topic1_PMIs), mean(topic2_PMIs)))
    
    if topic1_PMIs!=[]: pmi_refuting_dict[target]=mean(topic1_PMIs)
    if topic2_PMIs!=[]: pmi_discussion_dict[target]=mean(topic2_PMIs)

refuting_dict=Counter(pmi_refuting_dict)
discussion_dict=Counter(pmi_discussion_dict)
print('=====Refuting Key Words====')
print(refuting_dict.most_common(10))
print('=====Discussion Key Words====')
print(discussion_dict.most_common(10))
#print(co_counts['emigrate']['discuss'] )
