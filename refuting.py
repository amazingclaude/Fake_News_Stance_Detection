#======================================================
#===================refuting.py==============
#======================================================

from clean import clean
def mutual_information_title(string):
    _key_words = [
     
       
        #reportedly
        #'reportedly'
        'according',
        'said',
        'reported',
        'told',
        #claim
        #'claim'
        'claimed',
        'said',
        'would',
        'false',

        #hoax
        #'hoax'
        'culkin',
        'macaulay',
        'internet',
        'story',
        #fake
        #'fake'
        'facebook',
        'site',
        'real',
        'website',
        

        
      
    ]
    X = []
    
    clean_headline = clean(string)
    features = [1 if word in clean_headline else 0 for word in _key_words]
    X.append(features)
    return X

def mutual_information_body(string):
    _key_words = [
         #reportedly
        #'reportedly'
        'according',
        'said',
        'reported',
        'told',
        #claim
        #'claim'
        'claimed',
        'said',
        'would',
        'false',

        #hoax
        #'hoax'
        'culkin',
        'macaulay',
        'internet',
        'story',
        #fake
        #'fake'
        'facebook',
        'site',
        'real',
        'website',
       
        
    ]
    X = []
    
    clean_headline = clean(string)
    features = [1 if word in clean_headline else 0 for word in _key_words]
    X.append(features)
    return X

def refuting_features_title(string):
    _refuting_words = [
        #refuting words
        'fake',
        'fraud',
        'hoax','hoaxer',
        'false',
        'deny', 'denies',
        'despite',
        'nope',
        'doubt', 
        'bogus',
        'debunk',
        'pranks',
        'retract',  
        'lie',
        
        'not',
        'no',
        'didn',
        #discussion words
        'reportedly','report',
        'likely',
        'probably',
        'according',
        'might',
        
        'said',
        #key word 
        'update'
        #question mark
        '?',
      
    ]
    X = []
    
    clean_headline = clean(string)
    features = [1 if word in clean_headline else 0 for word in _refuting_words]
    X.append(features)
    return X

def refuting_features_body(string):
    _refuting_words = [
        #refuting words
        'fake',
        'fraud',
        'hoax','hoaxer',
        'false',
        'deny', 'denies',
        'despite',
        'nope',
        'doubt', 
        'bogus',
        'debunk',
        'pranks',
        'retract',
        'lie',
        
        #discussion words
        'reportedly','report',
        'likely',
        'probably',
        'according',
        'might',
        #key word 
        'update'
       
    ]
    X = []
    
    clean_headline = clean(string)
    features = [1 if word in clean_headline else 0 for word in _refuting_words]
    X.append(features)
    return X
 