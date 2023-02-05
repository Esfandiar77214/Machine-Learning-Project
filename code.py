####################### Request library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
import io
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split as tts

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout,Conv2D, MaxPool2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import Constant
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import mode
from IPython.display import display, HTML
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
import keras
import tensorflow as tf
import tensorflow.compat.v1 as tf
from keras.layers.convolutional import Conv1D    
from keras.layers import  Reshape
from keras.layers import Conv2D, MaxPooling2D,MaxPooling1D
from tensorflow.keras.utils import to_categorical
import pickle
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.multiclass import OutputCodeClassifier
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, MaxPool1D, Reshape, Activation
from sklearn.decomposition import PCA
from sklearn import svm
from keras.utils import np_utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from numpy import mean
from numpy import std
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.layers import concatenate
from keras.utils.vis_utils import plot_model
from numpy import dstack
from keras.layers import TimeDistributed

#################################Import Data
HAR = pd.read_csv(r'D:\University\Tehran University\Statistical Machine learning _ Dr.Amini\Project\Python\Data\time_series_data_human_activities.csv')

################################# Describe Data
HAR.info
print('Is in duplicates in HAR: {}'.format(sum(HAR.duplicated())))
print('We have {} Null values in HAR'.format(HAR.isnull().values.sum()))
print('We have {} NAN values in HAR'.format(HAR.isna().values.sum()))

# Define column name of the label vector
def gender_to_numeric(x):
    
        if x=='Walking':    return 1
        if x=='Jogging':    return 2
        if x=='Upstairs':   return 3
        if x=='Downstairs': return 4
        if x=='Sitting':    return 5
        if x=='Standing':   return 6

HAR['activity_num'] = HAR['activity'].apply(gender_to_numeric)
print(HAR)

#Define X and y
X = HAR[['x-axis','y-axis','z-axis']].values
X
X.shape
y = HAR[['activity_num']].values
y
y.shape

#Define Train and test dataset

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
#Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



######################SVM Model

# function to create classifier
def svm_fit(X,y,classification_type):
    model = svm.SVC(decision_function_shape= classification_type,kernel='linear')
    model.fit(X,y)
    e=model.score(X,y)
    return model,e

svm_fit(X,y,classification_type='ovr')
# function to computer mis-classification error
def error(model,X,y):
    e=model.score(X,y)
    return e
    
# function to assemble pipeline
def run_model(X_train,y_train,X_test,y_test,features):
    # building model
    model,e_train=svm_fit(X_train,y_train,'ovo')
    e_dev=error(model,X_train,y_train)
    e_test=error(model,X_test,y_test)
    
    print('Training accuracy=',e_train)
    print('Development accuracy=',e_dev)
    print('Testing accuracy=',e_test)
    
    cm=confusion_matrix(y_test,model.predict(X_test))
    return model,cm


###### Machine learning model

# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
# nonlinear models
    models['knn'] = KNeighborsClassifier(n_neighbors=7)
    models['cart'] = DecisionTreeClassifier()
    models['bayes'] = GaussianNB()
# ensemble models
    models['bag'] = BaggingClassifier(n_estimators=100)
    models['rf'] = RandomForestClassifier(n_estimators=100)
    models['et'] = ExtraTreesClassifier(n_estimators=100)
    models['gbm'] = GradientBoostingClassifier(n_estimators=100)
    print('Defined %d models' % len(models))
    return models

# evaluate a single model
def evaluate_model(X_train, y_train, X_test, y_test, model):
# fit the model
    model.fit(X_train, y_train)
# make predictions
    yhat = model.predict(X_test)
# evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    return accuracy * 100.0


# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
    results = dict()
    for name, model in models.items():
# evaluate the model
        results[name] = evaluate_model(X_train, y_train, y_test, y_test, model)
# show process
        print('>%s: %.3f' % (name, results[name]))
        return results
# print and plot the results
def summarize_results(results, maximize=True):
# create a list of (name, mean(scores)) tuples
    mean_scores = [(k,v) for k,v in results.items()]
# sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
# reverse for descending order (e.g. for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))
        print()
        for name, score in mean_scores:
            print('Name=%s, Score=%.3f' % (name, score))
# load dataset
# get model list
models = define_models()
# evaluate models
results = evaluate_models(X_train, y_train, X_test, y_test, models)
# summarize results
summarize_results(results)



##################Deep learning models
#run GPU
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.compat.v1.Session(config=config)

#####CNN


# zero-offset class values
y_train = y_train - 1
y_test = y_test - 1
# one hot encode y
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# fit and evaluate a model
def evaluate_model(X_train, y_train, X_test, y_test):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[1], y_train.shape[1]
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
    accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    return accuracy
evaluate_model(X_train, y_train, X_test, y_test)
# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
# run an experiment
def run_experiment(repeats=10):
# repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train, y_train, X_test, y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
# summarize results
    summarize_results(scores)
# run the experiment
run_experiment()



#########Multi - CNN Model

y_train = y_train - 1
y_test = y_test - 1
# one hot encode y
trainy = to_categorical(y_train)
testy = to_categorical(y_test)
# fit and evaluate a model
def evaluate_model(X_train, y_train, X_test, y_test):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
# head 1
    inputs1 = Input(shape=(n_timesteps,n_features))
    conv1 = Conv1D(64, 3, activation='relu')(inputs1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D()(drop1)
    flat1 = Flatten()(pool1)
# head 2
    inputs2 = Input(shape=(n_timesteps,n_features))
    conv2 = Conv1D(64, 5, activation='relu')(inputs2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D()(drop2)
    flat2 = Flatten()(pool2)
# head 3
    inputs3 = Input(shape=(n_timesteps,n_features))
    conv3 = Conv1D(64, 11, activation='relu')(inputs3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D()(drop3)
    flat3 = Flatten()(pool3)
# merge
    merged = concatenate([flat1, flat2, flat3])
# interpretation
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
# save a plot of the model
    plot_model(model, show_shapes=True, to_file='multiheaded.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
    model.fit([X_train,X_train,X_train], y_train, epochs=epochs, batch_size=batch_size,
    verbose=verbose)
# evaluate model
    accuracy = model.evaluate([X_test,X_test,X_test], y_test, batch_size=batch_size, verbose=0)
    return accuracy
# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
# run an experiment
def run_experiment(repeats=10):
# repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train, y_train, y_test, y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
# summarize results
    summarize_results(scores)
# run the experiment
run_experiment()


####################LSTM Model
# zero-offset class values
y_train = y_train - 1
y_test = y_test - 1
# one hot encode y
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# fit and evaluate a model
def evaluate_model(X_train, y_train, X_test, y_test):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
    accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    return accuracy
# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
# run an experiment
def run_experiment(repeats=10):
# repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train, y_train, X_test, y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
# summarize results
    summarize_results(scores)
# run the experiment
run_experiment()



##################CNN_LSTM Model

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
# define model
    verbose, epochs, batch_size = 0, 25, 64
    n_features, n_outputs = X_train.shape[2], y_train.shape[1]
# reshape data into time steps of sub-sequences
    n_steps, n_length = 4, 32
    trainX = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
    testX = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))
# define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(64, 3, activation='relu'),
        input_shape=(None,n_length,n_features)))
    model.add(TimeDistributed(Conv1D(64, 3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
    accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    return accuracy
# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
# run an experiment
def run_experiment(repeats=10):
# repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train, y_train, X_test, y_test)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
# summarize results
    summarize_results(scores)
# run the experiment
run_experiment()




