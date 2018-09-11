#======================================================
#===================pred_train.py==============
#======================================================

#Adapted from https://github.com/uclmr/fakenewschallenge/blob/master/util.py
#Original credit - @jaminriedel 

# Import relevant packages and modules
from util import FNCData,bow_train,pipeline_train,pipeline_test,save_predictions
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from score import report_score
from split import split_training_data
import os

# Prompt for mode
mode = input('mode (load / train)? ')

# Initialise hyperparameters
r = random.Random()
r.seed(123)
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 512 #batch number must be smaller than training samples, otherwise the training process will be skiped.
epochs = 90
base_dir='split-data'

#=====================
# Define model
#=====================
def model(dataset_number,base_dir,iter):
    # Process data sets  
    train_set, train_stances = pipeline_train(dataset_number,raw_train,bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    test_set = pipeline_test(dataset_number,raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)      
    feature_size = len(train_set[0])
    # Define Graph
    #Clear the graph (Closing session does not reset graph by design.The number of nodes available in the current graph will keep increasing if not reset)
    tf.reset_default_graph()
    # Create placeholders
    features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
    stances_pl = tf.placeholder(tf.int64, [None], 'stances')
    keep_prob_pl = tf.placeholder(tf.float32)

    # Infer batch size
    batch_size = tf.shape(features_pl)[0]

    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.contrib.layers.fully_connected(features_pl, hidden_size,weights_initializer=tf.contrib.layers.xavier_initializer(seed=100)), seed=101, keep_prob=keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size,weights_initializer=tf.contrib.layers.xavier_initializer(seed=102)),seed=103,keep_prob=keep_prob_pl)
    logits = tf.reshape(logits_flat, [batch_size, target_size])#reshape to be (batch_size*4)

    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

    # Define overall loss
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl) + l2_loss)

    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    #predict = tf.argmax(softmaxed_logits, 1)
    predict=softmaxed_logits
    # Define optimiser
    opt_func = tf.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    

    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))
    
    #Epoch_loss=[]
    # Perform training
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(epochs):
            total_loss = 0
            indices = list(range(n_train))
            r.shuffle(indices)#when conducting a seris of experiment, the shuffle will make the result different according to the sequence

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_features = [train_set[i] for i in batch_indices]
                batch_stances = [train_stances[i] for i in batch_indices]

                batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
                _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                total_loss += current_loss
            #print('epoch',epoch,'loss',total_loss)
            #Epoch_loss.append(total_loss)
            
        #Save the loss for comparison
        #save_loss=np.asarray(Epoch_loss)
        #np.savetxt("Epoch_loss.csv",save_loss,delimiter=",")
        #Save the model as checkpoints for re-store's purpose

        saver.save(sess, base_dir+'/model/model%d/mymodel%d'%(dataset_number,iter))
        
        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)       
        return test_pred,total_loss
#======================
# Restore model 
#======================
    
def restore_model(model_num,base_dir,iter):
    
    test_set = pipeline_test(model_num,raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    feature_size = len(test_set[0])
    print('============test set generated========================')
    # Define graph
    # Clear the graph (Closing session does not reset graph by design.The number of nodes available in the current graph will keep increasing if not reset)
    tf.reset_default_graph()
    # Create placeholders
    features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
    stances_pl = tf.placeholder(tf.int64, [None], 'stances')
    keep_prob_pl = tf.placeholder(tf.float32)

    # Infer batch size
   
    batch_size = tf.shape(features_pl)[0]

    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
    logits = tf.reshape(logits_flat, [batch_size, target_size])

    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

    # Define overall loss
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl) + l2_loss)

    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    predict = softmaxed_logits
    saver = tf.train.Saver()
    with tf.Session() as sess:
                
        
        saver.restore(sess,base_dir+'/model/model%d/mymodel%d'%(model_num,iter))

        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)

    return test_pred


       
# Load model
if mode == 'load':
    Grade=[]
    Agree=[]
    Disagree=[]
    Discuss=[]
    Unrelated=[]
    Recall=[]	
    k_fold=10
    #K-fold training,
    circle_time=k_fold#change circle_time to k_fold when doing cross validation
    for i in range (0,circle_time): 
        print('iteration:',i)
        fold_num=i
        split_training_data(fold_num,k_fold,base_dir)
        
        # Set file names
        file_train_instances = "train_stances.csv"
        file_train_bodies = "train_bodies.csv"
        file_test_instances = "test_stances_unlabeled.csv"
        file_test_bodies = "test_bodies.csv"
        file_predictions = 'predictions_test.csv'
       
       
        # Load data sets with base_dir
        raw_train = FNCData(base_dir,file_train_instances, file_train_bodies)
        raw_test = FNCData(base_dir,file_test_instances, file_test_bodies)    
        n_train = len(raw_train.instances)#the total number of entry instances
        bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = bow_train(raw_train, raw_test, lim_unigram=lim_unigram)
        
        weight_pred_1=np.diag(np.ones(4))
        #weight_pred_1[0][0]=2
        #weight_pred_1[1][1]=2
        weight_pred_2=np.diag(np.ones(4))
        #weight_pred_2[2][2]=2
        weight_pred_3=np.diag(np.ones(4))
        #weight_pred_3[3][3]=2
        
        
        test_prediction1=restore_model(1,base_dir,fold_num)
        #print('======================first model finished==========')
        #test_prediction2=restore_model(3,base_dir,fold_num) 
        #print('======================second model finished==========')
        #test_prediction3=restore_model(3,base_dir,fold_num) 
        
        #====ensemble for two(weighted)====
        #final_pred=np.concatenate((np.matmul(test_prediction1,weight_pred_1),np.matmul(test_prediction2,weight_pred_2)),axis=1)
        #final_pred=np.matmul(test_prediction1,weight_pred_1)+np.matmul(test_prediction2,weight_pred_2)


        #=======ensemble for three=========
        #final_pred=np.concatenate((np.matmul(test_prediction1,weight_pred_1),np.matmul(test_prediction2,weight_pred_2),np.matmul(test_prediction3,weight_pred_3)),axis=1)
        #final_pred=np.matmul(test_prediction1,weight_pred_1)+np.matmul(test_prediction2,weight_pred_2)+np.matmul(test_prediction3,weight_pred_3)
        #=======no ensemble ========
        final_pred=test_prediction1 
        #====================================
        final_pred_index = np.argmax(final_pred,1) 
        save_predictions(base_dir,final_pred_index, file_predictions)
        # =========Calculate score=================================================
        golden_stance = pd.read_csv(base_dir+"/"+"test_stances_labeled.csv")
        prediction_stance=pd.read_csv(base_dir+"/"+"predictions_test.csv")
        competition_grade,agree_recall,disagree_recall,discuss_recall,unrelated_recall,all_recall=report_score(golden_stance['Stance'],prediction_stance['Prediction'])
        
        Grade.append(competition_grade)
        Agree.append(agree_recall)
        Disagree.append(disagree_recall)
        Discuss.append(discuss_recall)
        Unrelated.append(unrelated_recall)
        Recall.append(all_recall)
        
        
        print('Grade',Grade)
        print('Agree',Agree)
        print('Disagree',Disagree)
        print('Discuss',Discuss)
        print('Unrelated',Unrelated)
        print('All Recall',Recall)

        print('mean Grade',np.mean(Grade))
        print('mean Agree',np.mean(Agree))
        print('mean Recall',np.mean(Recall))
    # Save the k-fold performance to csv
    df = pd.DataFrame({"Grade" : np.array(Grade), "Agree" : np.array(Agree),"Disagree" : np.array(Disagree),"Discuss" : np.array(Discuss),"Unrelated" : np.array(Unrelated),"Recall" : np.array(Recall)})
    df.to_csv(base_dir+'/'+"Performance.csv", index=False)

if mode == 'train':
    Grade=[]
    Agree=[]
    Disagree=[]
    Discuss=[]
    Unrelated=[]
    Recall=[]	
    Loss=[]
    k_fold=10
    #K-fold training,
    circle_time=k_fold#change circle_time to k_fold when doing cross validation
    for i in range (0,circle_time): 
        print('iteration:',i)
        fold_num=i
        split_training_data(fold_num,k_fold,base_dir)
        
        # Set file names
        file_train_instances = "train_stances.csv"
        file_train_bodies = "train_bodies.csv"
        file_test_instances = "test_stances_unlabeled.csv"
        file_test_bodies = "test_bodies.csv"
        file_predictions = 'predictions_test.csv'
       
       
        # Load data sets with base_dir
        raw_train = FNCData(base_dir,file_train_instances, file_train_bodies)
        raw_test = FNCData(base_dir,file_test_instances, file_test_bodies)    
        n_train = len(raw_train.instances)#the total number of entry instances
        
        bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = bow_train(raw_train, raw_test, lim_unigram=lim_unigram)
       
      
        # dataset=1: baseline feature [head_tf, body_tf, tfidf_cos]
        # dataset=2: baseline+refuting words feature
        # dataset=3: baseline+mutual information words feature
        # dataset=4: baseline+word2vec similarity feature
        # dataset=5: baseline+wmd distance feature
        # dataset=6: baseline+combining feature from 2 to 5

        
        test_prediction1,total_loss=model(3,base_dir,fold_num) 
        #test_prediction2,total_loss=model(2,base_dir,fold_num) 
        #test_prediction3,total_loss=model(6,base_dir) 
        
        weight_pred_1=np.diag(np.ones(4))
        #weight_pred_1[0][0]=2
        #weight_pred_1[1][1]=2
        weight_pred_2=np.diag(np.ones(4))
        #weight_pred_2[2][2]=2
        weight_pred_3=np.diag(np.ones(4))
        #weight_pred_3[3][3]=2 
        #===========ensemble===================
        #final_pred=np.concatenate((np.matmul(test_prediction1,weight_pred_1),np.matmul(test_prediction2,weight_pred_2)),axis=1)

        #final_pred=np.concatenate((np.matmul(test_prediction1,weight_pred_1),np.matmul(test_prediction2,weight_pred_2),np.matmul(test_prediction3,weight_pred_3)),axis=1)
        #final_pred=test_prediction1+test_prediction2+test_prediction3
        #========================================
        final_pred=test_prediction1
        final_pred_index = np.argmax(final_pred,1)
        
        # Save predictions
        save_predictions(base_dir,final_pred_index, file_predictions)
        # Calculate score
        golden_stance = pd.read_csv(base_dir+"/"+"test_stances_labeled.csv")
        prediction_stance=pd.read_csv(base_dir+"/"+"predictions_test.csv")
        competition_grade,agree_recall,disagree_recall,discuss_recall,unrelated_recall,all_recall=report_score(golden_stance['Stance'],prediction_stance['Prediction'])
        
        Grade.append(competition_grade)
        Agree.append(agree_recall)
        Disagree.append(disagree_recall)
        Discuss.append(discuss_recall)
        Unrelated.append(unrelated_recall)
        Recall.append(all_recall)
        Loss.append(total_loss)
        # Save the k-fold performance to csv
        df = pd.DataFrame({"Grade" : np.array(Grade), "Agree" : np.array(Agree),"Disagree" : np.array(Disagree),"Discuss" : np.array(Discuss),"Unrelated" : np.array(Unrelated),"Recall" : np.array(Recall),"Loss" : np.array(Loss),})
        df.to_csv(base_dir+'/'+"Performance.csv", index=False)
        
        print('Grade',Grade)
        print('Agree',Agree)
        print('Disagree',Disagree)
        print('Discuss',Discuss)
        print('Unrelated',Unrelated)
        print('All Recall',Recall)
        print('Total Loss',Loss)

        print('mean Grade',np.mean(Grade))
        print('mean Agree',np.mean(Agree))
        print('mean Recall',np.mean(Recall))