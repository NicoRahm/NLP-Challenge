import random
import numpy as np
import pickle

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.activations import softmax, sigmoid


import math

import xgboost as xgb

#selected_features = range(23)
selected_features = [0,1,2,3,4,5,6,7,8,9,11,13,15,17]

# Model
#==============================================================================

gbm = xgb.XGBClassifier(max_depth=3, 
                        n_estimators=200, 
                        learning_rate=0.1,
                        reg_alpha=0.001)


model = Sequential()
model.add(Dense(32, input_shape = (len(selected_features),)))
model.add(Activation('sigmoid'))

model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))

model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))

model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'Adam', loss = 'binary_crossentropy')


#==============================================================================


fileObject = open("data/saved_data2",'rb')  
saved = pickle.load(fileObject)
fileObject.close()


#Testing
#==============================================================================

#data = saved["training_features"]
#labels_array = saved["training_labels"]
# 
#to_keep_train = range(int(len(data)*0.95))
#to_keep_test = range(int(len(data)*0.95), len(data))
#
#training_features = data[to_keep_train]
#labels_array_train = labels_array[to_keep_train]
#testing_features = data[to_keep_test]
#labels_array_test = labels_array[to_keep_test]
#
#training_features = training_features[:, selected_features]
#testing_features = testing_features[:, selected_features]
#
#training_features = scale(training_features)
#testing_features = scale(testing_features)
#
#gbm.fit(training_features, labels_array_train)
#predictions_XGB = gbm.predict(testing_features)
#print("f1 Score XGB : ", f1_score(y_true=labels_array_test, y_pred = np.round(predictions_XGB)))
#
#model.fit(training_features, labels_array_train, nb_epoch = 4, batch_size = 258)
#predictions_DL = model.predict_classes(testing_features)
#print("f1 Score DL : ", f1_score(y_true=labels_array_test, y_pred = np.round(predictions_DL)))

#==============================================================================

# Issue prediction 
#==============================================================================
training_features = saved["training_features"]
labels_array = saved["training_labels"]
testing_features = saved["testing_features"]

training_features = scale(training_features)
testing_features = scale(testing_features)

training_features = training_features[:, selected_features]
testing_features = testing_features[:, selected_features]

gbm.fit(training_features, labels_array)
predictions = gbm.predict(testing_features)

#model.fit(training_features, labels_array, nb_epoch = 1, batch_size = 512)
#predictions = model.predict_classes(testing_features)

f = open("improved_predictions.csv", 'w')
f.write('id,category\n')
for i in range(len(predictions)):
    f.write(str(i))
    f.write(',')
    f.write(str(predictions[i]))
    f.write('\n') 
f.close()

#==============================================================================
