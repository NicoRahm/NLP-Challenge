import random
import numpy as np
import pickle

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
from sklearn.metrics import f1_score

import math

import xgboost as xgb



fileObject = open("data/saved_data",'rb')  
saved = pickle.load(fileObject)
fileObject.close()

##Load
training_features = saved["training_features"]
labels_array = saved["training_labels"]

to_keep_train = range(int(len(training_features)*0.95))
to_keep_test = range(int(len(training_features)*0.95), len(training_features))
#
## Select data
training_features_train = training_features[to_keep_train]
labels_array_train = labels_array[to_keep_train]

training_features_train = training_features_train[:, [0,1,2,4,5,6,8,11]]


# initialize basic SVM
#param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
#dtrain = xgb.DMatrix(training_features, label=labels_array)
#bst = xgb.train(param, dtrain, 10)
gbm = xgb.XGBClassifier(max_depth=10, 
                        n_estimators=400, 
                        learning_rate=0.08,
                        reg_alpha = 0.01).fit(training_features_train, labels_array_train)
#classifier = svm.LinearSVC()

# train
#classifier.fit(training_features, labels_array)


# test
# Load

#testing_features = saved["testing_features"]


testing_features = training_features[to_keep_test]
testing_features = testing_features[:, [0,1,2,4,5,6,8,11]]

#
## Select data
#testing_features = testing_features[:, [0,1,2,3,4,6,7,8,9,10,11,12]]

# issue predictions
#predictions_SVM = list(classifier.predict(testing_features))
#dtest = xgb.DMatrix(testing_features)
#predictions_SVM = np.round(bst.predict(dtest))
predictions_SVM = gbm.predict(testing_features)

# Print F1 score
#labels_array = saved["testing_labels"]
#labels_array_test = labels_array[to_keep_test]
#print(predictions_SVM.shape)

#print("f1 Score : ", f1_score(y_true=labels_array_test, y_pred = predictions_SVM))


#write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
#predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

f = open("improved_predictions.csv", 'w')
f.write('id,category\n')
for i in range(len(predictions_SVM)):
    f.write(str(i))
    f.write(',')
    f.write(str(predictions_SVM[i]))
    f.write('\n') 
f.close()