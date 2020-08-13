
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
from keras.metrics import binary_crossentropy
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from mlp_utils import features_ext, dataset_split


def get_data(radius=2, nBits=256):
    
    print('Getting data')

    df = pd.read_csv('trainingData/dataset_single.csv')
    
    feat_list = list()
    for s in np.array(df.smiles):
        feat_list.append(features_ext(s))
    
    feat_arr = np.vstack(feat_list)

    data = np.concatenate((feat_arr, df.P1.values.reshape(-1, 1)), axis=1)
  
    return dataset_split(data, nBits) 


def training(X_train, y_train, X_test, y_test, **params):

    model = Sequential()
    model.add(Dense(256, input_shape=(X_train.shape[1], ), activation= "relu"))
    model.add(Dense(128, activation= "sigmoid"))
    model.add(Dense(64, activation= "sigmoid"))
    model.add(Dense(32, activation= "sigmoid"))
    model.add(Dense(16, activation= "sigmoid"))
    model.add(BatchNormalization(axis=1))
    model.add(Dense(1, activation= "sigmoid"))
    
    METRICS = [
#       keras.metrics.TruePositives(name='tp'),
#       keras.metrics.FalsePositives(name='fp'),
#       keras.metrics.TrueNegatives(name='tn'),
#       keras.metrics.FalseNegatives(name='fn'),
#       keras.metrics.BinaryAccuracy(name='accuracy'),
#       keras.metrics.Precision(name='precision'),
#       keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
    ]
    
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

    print(model.summary())
    
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
    
    no_claim_count, claim_count = np.bincount(y_train)
    total_count = len(y_train)
    weight_no_claim = (1 / no_claim_count) * (total_count) / 2.0
    weight_claim = (1 / claim_count) * (total_count) / 2.0
    class_weights = {0: weight_no_claim, 1: weight_claim}

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=params['epochs'], batch_size=10, verbose=1, callbacks=[es])
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("trainedModels/mlp_model.json", "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights("trainedModels/mlp_model.h5")
    print("Saved model to local space")
    
    y_pred = model.predict(X_test)
    print('Test auc:', roc_auc_score(y_test, y_pred) * 100)
    y_label = np.where(y_pred >= 0.5, 1, 0)   
    print('Test acc:', accuracy_score(y_test, y_label) * 100)
    print('Test f1:', f1_score(y_test, y_label) * 100)

# confusion Matrix measures : 
				#Test set:
				#AUC: 65%
				#F1 core: 88%
				#accuracy: 80%

if __name__ == '__main__':

    X_train, X_test, y_train, y_test = get_data(radius=2, nBits=256)
    
    params = dict(epochs=50)

    training(X_train, y_train, X_test, y_test, **params)
