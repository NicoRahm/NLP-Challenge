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
from preprocessor import preprocess


with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

###################
# random baseline #
###################

random_predictions = np.random.choice([0, 1], size=len(testing_set))
random_predictions = zip(range(len(testing_set)),random_predictions)

#with open("random_predictions.csv","wb") as pred:
#    csv_out = csv.writer(pred)
#    for row in random_predictions:
#        csv_out.writerow(row)
        
# note: Kaggle requires that you add "ID" and "category" column headers

###############################
# beating the random baseline #
###############################

# the following script gets an F1 score of approximately 0.66

# data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)

training_set = [element[0].split(" ") for element in training_set]

with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

IDs = [element[0] for element in node_info]

# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)

## the following shows how to construct a graph with igraph
## even though in this baseline we don't use it
## look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas

#edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]

## some nodes may not be connected to any other node
## hence the need to create the nodes of the graph from node_info.csv,
## not just from the edge list

#nodes = IDs

## create empty directed graph
#g = igraph.Graph(directed=True)
 
## add vertices
#g.add_vertices(nodes)
 
## add edges
#g.add_edges(edges)

# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set

# randomly select 5% of training set
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.003)))
training_set_reduced = [training_set[i] for i in to_keep]



# TF_IDF
tf_idf = []

# TF_IDF similarity 
tf_idf_sim = []

# Computing the TF_IDF
# Computing the TF_IDF

print("Storing terms from training documents as list of lists")
terms_by_doc = [document[5].split(" ") for document in node_info]
n_terms_per_doc = [len(terms) for terms in terms_by_doc]

# store all terms in list
all_terms = [terms for sublist in terms_by_doc for terms in sublist]

# compute average number of terms
avg_len = sum(n_terms_per_doc)/len(n_terms_per_doc)

# unique terms
all_unique_terms = list(set(all_terms))

print("Computing IDF values")
# store IDF values in dictionary
n_doc = len(training_set_reduced)

idf = dict(zip(all_unique_terms,[0]*len(all_unique_terms)))
counter = 0

for element in idf.keys():
    # number of documents in which each term appears
    df = sum([element in terms for terms in terms_by_doc])
    # idf
    idf[element] = math.log10(float(n_doc+1)/df)

    counter+=1
    if counter % 200 == 0:
        print(counter, "terms have been processed")

counter = 0
len_all = len(all_unique_terms)        
for i in range(len(terms_by_doc)):
    terms_in_doc = terms_by_doc[i]
    doc_len = len(terms_in_doc)
    
    feature_row_tfidf = [0]*len_all
    
    for term in list(set(terms_in_doc)):
        # number of occurences of word in document
        index = all_unique_terms.index(term)
        tf = terms_in_doc.count(term)
        idf_term = idf[term]

        # store TF-IDF value
        feature_row_tfidf[index] = ((1+math.log1p(1+math.log1p(tf)))/(1-0.2+(0.2*(float(doc_len)/avg_len)))) * idf_term
    
    
    tf_idf.append(feature_row_tfidf)
    counter+=1
    if counter % 500 == 0:
        print(counter, "documents have been processed")

training_features = preprocess(training_set_reduced, IDs, node_info)

# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)

# initialize basic SVM
classifier = svm.LinearSVC()

# train
classifier.fit(training_features, labels_array)


# test
# we need to compute the features for the testing set 
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.003)))
testing_set_reduced = [training_set[i] for i in to_keep]

testing_features = preprocess(testing_set_reduced, IDs, node_info)

# scale
testing_features = preprocessing.scale(testing_features)

# issue predictions
predictions_SVM = list(classifier.predict(testing_features))

# Print F1 score

labels = [int(element[2]) for element in testing_set_reduced]
labels = list(labels)
labels_array = np.array(labels)

print("f1 Score : ", f1_score(y_true=labels_array, y_pred = predictions_SVM))


# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
# predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

# with open("improved_predictions.csv","wb") as pred1:
#     csv_out = csv.writer(pred1)
#     for row in predictions_SVM:
#         csv_out.writerow(row)