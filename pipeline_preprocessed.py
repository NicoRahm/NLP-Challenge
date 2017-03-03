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

#Load
training_features = saved["training_features"]
labels_array = saved["training_labels"]

# initialize basic SVM
#param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
#dtrain = xgb.DMatrix(training_features, label=labels_array)
#bst = xgb.train(param, dtrain, 10)
gbm = xgb.XGBClassifier(max_depth=6, n_estimators=200, learning_rate=0.03).fit(training_features, labels_array)
#classifier = svm.LinearSVC()

# train
#classifier.fit(training_features, labels_array)


# test
# Load
testing_features = saved["testing_features"]


# issue predictions
#predictions_SVM = list(classifier.predict(testing_features))
#dtest = xgb.DMatrix(testing_features)
#predictions_SVM = np.round(bst.predict(dtest))
predictions_SVM = gbm.predict(testing_features)

# Print F1 score
labels_array = saved["testing_labels"]
print(predictions_SVM)

print("f1 Score : ", f1_score(y_true=labels_array, y_pred = predictions_SVM))


# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
# predictions_SVM = zip(range(len(testing_set)), predictions_SVM)

# with open("improved_predictions.csv","wb") as pred1:
#     csv_out = csv.writer(pred1)
#     for row in predictions_SVM:
#         csv_out.writerow(row)