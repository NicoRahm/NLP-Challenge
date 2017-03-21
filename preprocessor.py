import numpy as np
import nltk

import math

import os
import pickle

from library import *
from scipy.sparse import csr_matrix


nltk.download('punkt') # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

def preprocess(data_set_reduced, IDs, node_info, g_articles, degrees, closeness, g_authors, journals, journal_importance):
    # we will use three basic features:
    
    # number of overlapping words in title
    overlap_title = []
    
    # temporal distance between the papers
    temp_diff = []
    
    # number of common authors
    comm_auth = []
    
    # Source closeness
    source_close = []
    
    # Target degree
    target_deg = []
    
    # Importance of the principal author of the target 
    target_author_deg = []
    
    # Closeness of the principal author of the target
    target_author_close = []
    
    # Target journal importance 
    target_importance = []
    
    # Number of common article cited 
    comm_cit_OUT = []
    
    # Number of common article that cite it
    comm_cit_IN = []

    authors = [node[3].split(",") for node in node_info]
    unique_authors = list(set([item for sublist in authors for item in sublist]))
    
    
    counter = 0
    for i in range(len(data_set_reduced)):
        source = data_set_reduced[i][0]
        target = data_set_reduced[i][1]
        
        index_source = IDs.index(source)
        index_target = IDs.index(target)
        
        source_info = [element for element in node_info if element[0]==source][0]
        target_info = [element for element in node_info if element[0]==target][0]
        
        
        target_deg.append(degrees[index_target])
        source_close.append(closeness[index_source])
        
        if node_info[index_target][3].split(",")[0] != '':
            target_author_deg.append(g_authors.degree(unique_authors.index(node_info[index_target][3].split(",")[0])))
            target_author_close.append(g_authors.closeness(unique_authors.index(node_info[index_target][3].split(",")[0])))
        
        else:
            target_author_deg.append(0)
            target_author_close.append(0)
        
            
        if (node_info[index_target][4] != ''):
            target_importance.append(journal_importance[journals.index(node_info[index_target][4])])
        else:
            target_importance.append(0)
        
        source_neighb = set(g_articles.neighbors(index_source, mode = 'IN'))
        target_neighb = set(g_articles.neighbors(index_target, mode = 'IN'))
        
        comm_cit_IN.append(len(source_neighb.intersection(target_neighb)))
        
        source_neighb = set(g_articles.neighbors(index_source, mode = 'OUT'))
        target_neighb = set(g_articles.neighbors(index_target, mode = 'OUT'))
        
        comm_cit_OUT.append(len(source_neighb.intersection(target_neighb)))

        
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
            print(counter, "data examples processed", round(100*(counter/len(data_set_reduced)), 2), "%")
        
    # convert list of lists into array
    # documents as rows, unique words as columns (i.e., example as rows, features as columns)
    
    return np.array([overlap_title, temp_diff, comm_auth, source_close, target_deg, target_author_deg, target_author_close, target_importance, comm_cit_IN, comm_cit_OUT]).T

# create the idf matrix and all_unique_terms with training data
def init_tw_idf(training_set, node_info, feature):
    #We only keep the info of the papers that are implicated in the given training_set (which can be reduced)
    filtered_node_ids = []
    for link_data in training_set:
        filtered_node_ids.append(link_data[0])
        filtered_node_ids.append(link_data[1])
    #remove duplicate
    filtered_node_ids = list(set(filtered_node_ids))
    #keep only relevant papers
    node_info = np.array([node for node in node_info if node[0] in filtered_node_ids])
    
    #rows is meants for list of abstracts or list of titles
    rows = np.array([d[feature['index']] for d in node_info])
    
    print("Initializing TW_IDF with", len(training_set), "training documents ")
    terms_by_row = [row.split(" ") for row in rows]
    # store all terms in list
    all_terms = [terms for sublist in terms_by_row for terms in sublist]
    # unique terms
    all_unique_terms = list(set(all_terms))
    print("all_unique_terms : " , len(all_unique_terms))
    
    idf = dict(list(zip(all_unique_terms,[0]*len(all_unique_terms))))
    counter = 0
    n_rows = len(rows)
    for element in idf.keys():
        # number of documents in which each term appears
        df = sum([element in terms for terms in terms_by_row])
        # idf
        idf[element] = math.log10(float(n_rows+1)/df)
    
        counter+=1
        if counter % 200 == 0:
            print(counter, "terms have been processed", round(100*(counter/len(idf.keys())), 2), "%")
    return all_unique_terms, idf



def compute_text_metrics(node_info, feature, text_type):
    
    saver = {}
    
    all_unique_terms = feature['all_unique_terms']
    idf = feature['idf']
    
    abstracts = np.array([d[5] for d in node_info])
    
    terms_by_doc = [document.split(" ") for document in abstracts]
    n_terms_per_doc = [len(terms) for terms in terms_by_doc]
    # compute average number of terms
    avg_len = sum(n_terms_per_doc)/len(n_terms_per_doc)
    print("min, max and average number of terms per document:", min(n_terms_per_doc), max(n_terms_per_doc), avg_len)
    
    print("creating a graph-of-words for each document...")

    window = 3
    all_graphs = []
    for terms in terms_by_doc:
        all_graphs.append(terms_to_graph(terms,window))
    
    # sanity checks (should return True)
    print(len(terms_by_doc)==len(all_graphs))
    print(len(set(terms_by_doc[0]))==len(all_graphs[0].vs))
    
    print("computing vector representations of each document")
    
    b = 0.003
    k1= 0.05
    
    features_degree = []
    features_w_degree = []
    features_closeness = []
    features_w_closeness = []
    features_tfidf = []
    bm25 = []
    
    len_all = len(all_unique_terms)
    
    counter = 0
    
    for i in range(len(all_graphs)):
    
        graph = all_graphs[i]
        # retain only the terms originally present in the data test
        terms_in_doc = [term for term in terms_by_doc[i] if term in all_unique_terms]
        doc_len = len(terms_in_doc)
    
        # returns node (1) name, (2) degree, (3) weighted degree, (4) closeness, (5) weighted closeness
        my_metrics = compute_node_centrality(graph)
        feature_row_degree = [0]*len_all
        feature_row_w_degree = [0]*len_all
        feature_row_closeness = [0]*len_all
        feature_row_w_closeness = [0]*len_all
        feature_row_tfidf = [0]*len_all
        bm25_row = [0]*len_all
    
        for term in list(set(terms_in_doc)):
            index = all_unique_terms.index(term)
            idf_term = idf[term]
            denominator = (1-b+(b*(float(doc_len)/avg_len)))
            metrics_term = [t[1:5] for t in my_metrics if t[0]==term][0]
    
            # store TW-IDF values
            feature_row_degree[index] = (float(metrics_term[0])/denominator) * idf_term
            feature_row_w_degree[index] = (float(metrics_term[1])/denominator) * idf_term
            feature_row_closeness[index] = (float(metrics_term[2])/denominator) * idf_term
            feature_row_w_closeness[index] = (float(metrics_term[3])/denominator) * idf_term
    
            # number of occurences of word in document
            tf = terms_in_doc.count(term)
    
            # store TF-IDF value
            feature_row_tfidf[index] = ((1+math.log1p(1+math.log1p(tf)))/(1-0.2+(0.2*(float(doc_len)/avg_len)))) * idf_term
            bm25_row[index] = idf_term * (k1 +1)*tf/(k1 + tf)
    
        features_degree.append(csr_matrix(feature_row_degree))
        features_w_degree.append(csr_matrix(feature_row_w_degree))
        features_closeness.append(csr_matrix(feature_row_closeness))
        features_w_closeness.append(csr_matrix(feature_row_w_closeness))
        features_tfidf.append(csr_matrix(feature_row_tfidf))
        bm25.append(csr_matrix(bm25_row))
    
        counter += 1
        if counter % 200 == 0:
            print(counter, "documents have been processed", round(100*(counter/len(all_graphs)), 2), "%")
    
    saver["degree"] = features_degree
    saver["w_degree"] = features_w_degree
    saver["closeness"] = features_closeness
    saver["w_closeness"] = features_w_closeness
    saver["tfidf"] = features_tfidf
    saver["bm25"] = bm25
         
    file_Name = "data/text_metrics_" + text_type 
    fileObject = open(file_Name,'wb') 
    pickle.dump(saver,fileObject) 
    fileObject.close()
    
    return(features_degree, features_w_degree, features_closeness, features_w_closeness, features_tfidf, bm25)

    
    

def add_tw_idf(data_features, data_set, node_info, feature, g_cit, text_type):
    
    path_init = "data/text_metrics_" + text_type
    if os.path.isfile(path_init):
        fileObject = open(path_init,'rb') 
        s = pickle.load(fileObject)
        features_degree = s["degree"]
        features_closeness = s["closeness"]
        features_w_closeness = s["w_closeness"]
        features_w_degree = s["w_degree"]
        features_tfidf = s["tfidf"]
        bm25 = s["bm25"]
        print("Text metrics loaded for the " + text_type)
        
    else: 
        features_degree, features_w_degree, features_closeness, features_w_closeness, features_tfidf, bm25 = compute_text_metrics(node_info, feature, text_type)
        
    abstracts_index = np.array([d[0] for d in node_info])  

    # Now we have a representation of each abstract (in fact 5 different representations !)
    # We'll find the pairs associated to each possible link et calculate the "produit scalaire"
    twidf_features = []
    counter = 1
    for link_data in data_set:
        
        index1 = np.where(abstracts_index == link_data[0])[0][0]
        index2 = np.where(abstracts_index == link_data[1])[0][0]
        
        tfidf_citations = 0
        degree_citations = 0
        c = 0
        neigh = g_cit.neighbors(index1, mode = 'OUT')
        for n in neigh: 
            if(index2 != n):
                c+=1
                tfidf_citations += features_tfidf[index2].dot(features_tfidf[n].transpose()).toarray()[0][0]
                degree_citations += features_degree[index2].dot(features_degree[n].transpose()).toarray()[0][0]
        if(c != 0):
            tfidf_citations/=c
            degree_citations/=c
        
        tfidf_product = features_tfidf[index1].dot(features_tfidf[index2].transpose()).toarray()[0][0]
        
        degree_product = features_degree[index1].dot(features_degree[index2].transpose()).toarray()[0][0]
        
        w_degree_product = features_w_degree[index1].dot(features_w_degree[index2].transpose()).toarray()[0][0]
        
        closeness_product = features_closeness[index1].dot(features_closeness[index2].transpose()).toarray()[0][0]
        
        w_closeness_product = features_w_closeness[index1].dot(features_w_closeness[index2].transpose()).toarray()[0][0]
        
        source = bm25[index1].toarray()
        target = bm25[index2].toarray()
        BM25 = source[:,target[0] != 0].sum()
        
        twidf_features.append([tfidf_product, degree_product, w_degree_product, closeness_product,w_closeness_product, BM25, tfidf_citations, degree_citations])
        
        counter += 1
        if counter % 500 == 0:
            print(counter, "data exemples have been processed for the " + text_type , round(100*(counter/len(data_set)), 2), "%")
        
    #outlook for curiosity
    print("twidf_features", twidf_features[:2])
    
    new_data_features = []
    for i in range(len(data_features)):
            new_data_features.append(np.append(data_features[i], twidf_features[i]))
            #if i < 2:
            #    print(new_data_features[i])
    print("tw_idf successful")
    return np.array(new_data_features)
    