import random
import numpy as np
import pickle
import os.path

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
from preprocessor import *

saver = {}
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

#Outlook on the data
#print(training_set[:5])
#print(node_info[10000:10001])

IDs = [element[0] for element in node_info]

# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)




# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set

# randomly select 5% of training set
ratio = 0.003
print("RATIO :", ratio)
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*ratio)))
training_set_reduced = [training_set[i] for i in to_keep]


print("Computing the graph of document citation")

# Construct the graph of the citation
edges = [(element[0],element[1]) for element in training_set_reduced if element[2]=="1"]

nodes = IDs

# create empty directed graph
g = igraph.Graph(directed=True)
 
# add vertices
g.add_vertices(nodes)
 
# add edges
g.add_edges(edges)

closeness = g.closeness(normalized=True)
closeness = [round(value,5) for value in closeness]
    
degrees = g.degree()
degrees = [round(float(degree)/(len(g.vs)-1),5) for degree in degrees]


print("Computing the graph of authors")

authors = [node[3].split(",") for node in node_info]
unique_authors = list(set([item for sublist in authors for item in sublist]))

edges = [(node_info[IDs.index(element[0])][3].split(",")[0], node_info[IDs.index(element[1])][3].split(",")[0]) for element in training_set_reduced if element[2]=="1" and node_info[IDs.index(element[0])][3].split(",")[0] != '' and node_info[IDs.index(element[1])][3].split(",")[0] != '']

# create empty directed graph
g_authors = igraph.Graph(directed=True)
 
# add vertices
g_authors.add_vertices(unique_authors)
 
# add edges
g_authors.add_edges(edges)

print("Computing importance of journal")

journals = list(set([node[4] for node in node_info if node[4] != '']))
journal_importance = [0]*len(journals)

for i in range(len(training_set_reduced)):
    target_journal = node_info[IDs.index(training_set_reduced[i][1])][4]
    if (target_journal != ''):
        journal_importance[journals.index(target_journal)] += 1

n_papers = [0]*len(journals)
for i in range(len(node_info)):
    journal = node_info[i][4]
    if (journal != ''):
        n_papers[journals.index(journal)] += 1

for i in range(len(journal_importance)):
    journal_importance[i]/= n_papers[i]

training_features = preprocess(training_set_reduced, IDs, node_info, degrees, closeness, g_authors, journals, journal_importance)

#Add tw-idf on abstracts
path_init = "data/idf_init_titles"
if os.path.isfile(path_init):
    print('IDF TITLES LOADED FROM FILE') 
    fileObject = open(path_init,'rb')  
    s = pickle.load(fileObject)
    fileObject.close()
    titles = {'index': 2} #index in column of node_info
    titles['all_unique_terms'] = s['all_unique_terms']
    titles['idf'] = s['idf']
else:
    print('IDF TITLES CALCULATED')
    titles = {'index': 2} #index in column of node_info
    titles['all_unique_terms'], titles['idf'] = init_tw_idf(training_set, node_info, titles)
    file_Name = path_init
    fileObject = open(file_Name,'wb') 
    s = {'all_unique_terms': titles['all_unique_terms'], 'idf':titles['idf']}
    pickle.dump(s,fileObject)
    fileObject.close()
    
path_init = "data/idf_init_abstracts"
if os.path.isfile(path_init):
    print('IDF ABSTRACTS LOADED FROM FILE') 
    fileObject = open(path_init,'rb')  
    s = pickle.load(fileObject)
    fileObject.close()
    abstracts = {'index': 2} #index in column of node_info
    abstracts['all_unique_terms'] = s['all_unique_terms']
    abstracts['idf'] = s['idf']
else:
    print('IDF ABSTRACTS CALCULATED')
    abstracts = {'index' : 5} #index in column of node_info    
    abstracts['all_unique_terms'], abstracts['idf'] = init_tw_idf(training_set_reduced, node_info, abstracts)
    file_Name = path_init
    fileObject = open(file_Name,'wb') 
    s = {'all_unique_terms': titles['all_unique_terms'], 'idf':titles['idf']}
    pickle.dump(s,fileObject)
    fileObject.close()

#abstracts = {'index' : 5} #index in column of node_info    
#abstracts['all_unique_terms'], abstracts['idf'] = init_tw_idf(training_set_reduced, node_info, abstracts)
#titles = {'index': 2} #index in column of node_info
#titles['all_unique_terms'], titles['idf'] = init_tw_idf(training_set_reduced, node_info, titles)

training_features = add_tw_idf(training_features, training_set_reduced, node_info, abstracts)
training_features = add_tw_idf(training_features, training_set_reduced, node_info, titles)

# scale
training_features = preprocessing.scale(training_features)

#Save
saver["training_features"] = training_features

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)
saver["training_labels"] = labels_array

# initialize basic SVM
classifier = xgb.XGBClassifier(n_estimators = 100, 
                               learning_rate = 0.1, 
                               max_depth = 3)

# train
classifier.fit(training_features, labels_array)


# test
# we need to compute the features for the testing set 
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*ratio)))
testing_set_reduced = [training_set[i] for i in to_keep]

testing_features = preprocess(testing_set_reduced, IDs, node_info, degrees, closeness, g_authors, journals, journal_importance)

testing_features = add_tw_idf(testing_features, testing_set_reduced, node_info, abstracts)
testing_features = add_tw_idf(testing_features, testing_set_reduced, node_info, titles)

# scale
testing_features = preprocessing.scale(testing_features)

saver["testing_features"] = testing_features

# issue predictions
predictions_SVM = list(classifier.predict(testing_features))

# Print F1 score

labels = [int(element[2]) for element in testing_set_reduced]
labels = list(labels)
labels_array = np.array(labels)
saver["testing_labels"] = labels_array

print("f1 Score : ", f1_score(y_true=labels_array, y_pred = predictions_SVM))

file_Name = "data/saved_data"
fileObject = open(file_Name,'wb') 
pickle.dump(saver,fileObject)
fileObject.close()



# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
# predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

# with open("improved_predictions.csv","wb") as pred1:
#     csv_out = csv.writer(pred1)
#     for row in predictions_SVM:
#         csv_out.writerow(row)