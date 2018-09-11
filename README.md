# Configuration
The following libraries were used in the code implementation. The compatibility was not tested on other TensorFlow versions. 
*	Python 3.6.4
*	NumPy 1.14.3
*	Scikit -learn 0.18.1
*	TensorFlow (GPU version) 1.8.0
*	Pandas 0.20.1
*	Genism 3.4.0
*	Nltk 3.3.0
*	Scipy 0.19.0
*	Pyemd 0.5.1
# Instruction for running the code
## Installation
Download GoogleNews-vectors-negative300.bin.gz , Google’s pretrained word2vec model in to folder google_news. No other separate installation is required.
Reproducing the result for FNC-1 evaluation
Execute pred.py file in load mode. This entails the following actions:
1.	The testing dataset will be loaded class FNCData in util.py.
2.	Function restore_model in pred.py will be intrigued which loads pre-trained models from checkpoints which are stored in file model.
3.	The models are used to generate predictions separately, then ensembled for a final prediction.
4.	Last. The predictions are saved in prediction_test.csv file and the performance index like accuracy and FNC-1 grade are saved in Performance.csv. 
## Retrain the model
* To generate keywords with stance class based mutual information algorithm, open stance_mi.py and specify the training documents. The training documents should follow the FNC format where one document contains a title/body pair with stance labels and one document contains body ID and body text. Then simply execute the python file, a list of words will be displayed directly on the command window. The number of words displayed can be adjusted at the last line of the coding.
* To generate keywords with customised class based mutual information algorithm, open mutual_information.py and specify the training documents and customised topic words in method mutual_information. Then simply execute the python file, a list of words will be displayed directly on the command window. The number of words displayed can be adjusted at the last line of the coding.
* To generate keywords with PMI algorithm, open pmi.py and specify the training documents. Then in method PMI specify two lists of words. Each list towards to a semantic orientation. Then simply execute the python file, a list of words will be displayed directly on the command window. The number of words displayed can be adjusted at the last line of the coding.
After the keywords are obtained, input them into function mutual_information_title and mutual_information_body in file refuting.py. Last, specify the model type in pred.py and execute it with mode train. 

