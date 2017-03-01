import random
import numpy as np
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from sklearn.metrics import f1_score
import nltk
import csv
import math

import xgboost as xgb
from library import terms_to_graph


nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()


def preprocess(data_set_reduced, IDs, node_info):
    # we will use three basic features:
    
    # number of overlapping words in title
    overlap_title = []
    
    # temporal distance between the papers
    temp_diff = []
    
    # number of common authors
    comm_auth = []
    
    counter = 0
    for i in range(len(data_set_reduced)):
        source = data_set_reduced[i][0]
        target = data_set_reduced[i][1]
        
        index_source = IDs.index(source)
        index_target = IDs.index(target)
        
        source_info = [element for element in node_info if element[0]==source][0]
        target_info = [element for element in node_info if element[0]==target][0]
        
    	# convert to lowercase and tokenize
        source_title = source_info[2].lower().split(" ")
    	# remove stopwords
        source_title = [token for token in source_title if token not in stpwds]
        source_title = [stemmer.stem(token) for token in source_title]
        
        target_title = target_info[2].lower().split(" ")
        target_title = [token for token in target_title if token not in stpwds]
        target_title = [stemmer.stem(token) for token in target_title]
        
        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",")
        
        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        temp_diff.append(int(source_info[1]) - int(target_info[1]))
        comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
       
        counter += 1
        if counter % 500 == 0:
            print(counter, "data examples processed")
        
    # convert list of lists into array
    # documents as rows, unique words as columns (i.e., example as rows, features as columns)
    
    return np.array([overlap_title, temp_diff, comm_auth]).T