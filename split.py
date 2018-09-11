#Adapted from https://github.com/FakeNewsChallenge/fnc-1-baseline/blob/master/utils/generate_test_splits.py
#Original credit - @FakeNewsChallenge 

from csv import DictReader
from csv import DictWriter
import random
import os
from collections import defaultdict

class Dataset():
    def __init__(self, name="Competition_train", path="split-data"):
        self.path = path

        print("Reading dataset")
        bodies = name+"_bodies.csv"
        stances = name+"_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))



    def read(self,filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding='utf-8-sig') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows
        
def generate_hold_out_split (dataset, fold_num,k_fold, base_dir):
    r = random.Random()
    r.seed(1489215)

    article_ids = list(dataset.articles.keys())  # get a list of article ids
    r.shuffle(article_ids)  # and shuffle that list
    fold=dict()
    spliting_ratio=1/k_fold
    for i in range(k_fold):
        fold[i]=article_ids[int(i*spliting_ratio * len(article_ids)):int((i+1)*spliting_ratio * len(article_ids))]

    development_set_ids=fold[fold_num]
    training_set_ids=[item for item in article_ids if item not in development_set_ids ]
    #training_set_ids=set(article_ids)-set(fold[fold_num])
    
    # write the split body ids out to files for future use
    with open(base_dir+ "/"+ "training_set_ids%d.txt"%fold_num, "w+") as f:
        f.write("\n".join([str(id) for id in training_set_ids]))

    with open(base_dir+ "/"+ "development_set_ids%d.txt"%fold_num, "w+") as f:
        f.write("\n".join([str(id) for id in development_set_ids]))
   
   
          
def read_text_ids(file,base):
    ids = []
    with open(base+"/"+file,"r") as f:
        for line in f:
           ids.append(int(line))
        return ids

def development_set_split(dataset,  fold_num,k_fold, base_dir):
    #if not (os.path.exists(base_dir+ "/"+ "training_set_ids.txt")
    #        and os.path.exists(base_dir+ "/"+ "development_set_ids.txt")):
    generate_hold_out_split(dataset, fold_num,k_fold,base_dir)

    training_set_ids = read_text_ids("training_set_ids%d.txt"%fold_num, base_dir)
    development_set_ids = read_text_ids("development_set_ids%d.txt"%fold_num, base_dir)
    
    stances_training_set = []
    stances_development_set = []
    bodies_training_set=[]
    bodies_development_set = []
    for stance in dataset.stances:
        if stance['Body ID'] in development_set_ids:
            stances_development_set.append(stance)
        else:         
            stances_training_set.append(stance)      
        
    return stances_training_set,stances_development_set,training_set_ids,development_set_ids

def save_stances_file(stance_instances, filename,base_dir):
    with open(base_dir+'/'+filename, 'w',encoding='utf-8-sig') as csvfile:
        fieldnames = ['Headline','Body ID','Stance']
        writer = DictWriter(csvfile, fieldnames=fieldnames,lineterminator='\n')

        writer.writeheader()
        for instance in stance_instances:
            writer.writerow({'Headline': instance['Headline'],'Body ID':instance['Body ID'],'Stance':instance['Stance']})

def save_bodies_file(body_ids,dataset, filename,base_dir):
    with open(base_dir+'/'+filename, 'w',encoding='utf-8-sig') as csvfile:
        fieldnames = ['Body ID','articleBody']
        writer = DictWriter(csvfile, fieldnames=fieldnames,lineterminator='\n')

        writer.writeheader()
        for article in body_ids:
            writer.writerow({'Body ID': article,'articleBody':dataset.articles[article]})
def split_training_data(fold_num,k_fold,base_dir):   
    d=Dataset()    
    #open(base_dir+'/'+'training_set_ids.txt', 'w',encoding='utf-8-sig')
    #open(base_dir+'/'+'development_set_ids.txt', 'w',encoding='utf-8-sig')

    stances_training_set,stances_development_set,training_set_ids,development_set_ids=development_set_split(d,fold_num,k_fold, base_dir)
    save_stances_file(stances_training_set,"train_stances.csv",base_dir)
    save_stances_file(stances_development_set,"test_stances_unlabeled.csv",base_dir)
    save_bodies_file(training_set_ids, d,"train_bodies.csv",base_dir)
    save_bodies_file(development_set_ids,d, "test_bodies.csv",base_dir)









