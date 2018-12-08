import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
english_stemmer=nltk.stem.SnowballStemmer('english')

from sklearn.feature_selection.univariate_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor,LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import itertools

from matplotlib import cm
import sys
import os
import argparse
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import six
from abc import ABCMeta
from scipy import sparse
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.svm import LinearSVC

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K



get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')



data = pd.read_csv('data/Combined_News_DJIA.csv')
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']


theadlines = []
for row in range(0,len(train.index)):
    theadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))



bvect = CountVectorizer()
btrain = bvect.fit_transform(theadlines)
print(btrain.shape)



advvect = TfidfVectorizer( min_df=0.03, max_df=0.97, max_features = 200000, ngram_range = (2, 2))
advtrain = advvect.fit_transform(theadlines)



print(advtrain.shape)



advancedmodel = LogisticRegression()
advancedmodel = advancedmodel.fit(advtrain, train["Label"])



teshead = []
for row in range(0,len(test.index)):
    teshead.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advatest = advvect.transform(teshead)
preds2 = advancedmodel.predict(advatest)
acc2=accuracy_score(test['Label'], preds2)


print('Logic Regression accuracy: ', acc2)



advwords = advvect.get_feature_names()
advcoeffs = advancedmodel.coef_.tolist()[0]
advcoeffdf = pd.DataFrame({'Words' : advwords, 
                        'Coefficient' : advcoeffs})
advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
advcoeffdf.head(5)


advcoeffdf.tail(5)



advvect = TfidfVectorizer( min_df=0.03, max_df=0.2, max_features = 200000, ngram_range = (2, 2))
advtrain = advvect.fit_transform(theadlines)


print(advtrain.shape)



advancedmodel = MultinomialNB(alpha=0.0001)
advancedmodel = advancedmodel.fit(advtrain, train["Label"])
teshead = []
for row in range(0,len(test.index)):
    teshead.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advatest = advvect.transform(teshead)
preds5 = advancedmodel.predict(advatest)
acc5 = accuracy_score(test['Label'], preds5)



print('NBayes accuracy: ', acc5)



class NBSVM(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):

    def __init__(self, alpha=1.0, C=1.0, max_iter=10000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.C = C
        self.svm_ = []

    def fit(self, X, y):
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        Y = Y.astype(np.float64)

        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.ratios_ = np.full((n_effective_classes, n_features), self.alpha,
                                 dtype=np.float64)
        self._compute_ratios(X, Y)

        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            svm = LinearSVC(C=self.C, max_iter=self.max_iter)
            Y_i = Y[:,i]
            svm.fit(X_i, Y_i)
            self.svm_.append(svm) 

        return self

    def predict(self, X):
        n_effective_classes = self.class_count_.shape[0]
        n_examples = X.shape[0]

        D = np.zeros((n_effective_classes, n_examples))

        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            D[i] = self.svm_[i].decision_function(X_i)
        
        return self.classes_[np.argmax(D, axis=0)]
        
    def _compute_ratios(self, X, Y):
        """Count feature occurrences and compute ratios."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")

        self.ratios_ += safe_sparse_dot(Y.T, X)
        normalize(self.ratios_, norm='l1', axis=1, copy=False)
        row_calc = lambda r: np.log(np.divide(r, (1 - r)))
        self.ratios_ = np.apply_along_axis(row_calc, axis=1, arr=self.ratios_)
        check_array(self.ratios_)
        self.ratios_ = sparse.csr_matrix(self.ratios_)




def f1_class(pred, truth, class_val):
    n = len(truth)

    truth_class = 0
    pred_class = 0
    tp = 0

    for ii in range(0, n):
        if truth[ii] == class_val:
            truth_class += 1
            if truth[ii] == pred[ii]:
                tp += 1
                pred_class += 1
                continue;
        if pred[ii] == class_val:
            pred_class += 1

    precision = tp / float(pred_class)
    recall = tp / float(truth_class)

    return (2.0 * precision * recall) / (precision + recall)


def semeval_senti_f1(pred, truth, pos=2, neg=0): 

    f1_pos = f1_class(pred, truth, pos)
    f1_neg = f1_class(pred, truth, neg)

    return (f1_pos + f1_neg) / 2.0;


def main(train_file, test_file, ngram=(1, 3)):
    print('loading...')
    train = pd.read_csv(train_file, delimiter='\t', encoding='utf-8', header=0,
                        names=['text', 'label'])


    test = pd.read_csv(test_file, delimiter='\t', encoding='utf-8', header=0,
                        names=['text', 'label'])

    print('vectorizing...')
    vect = CountVectorizer()
    classifier = NBSVM()

    clf = Pipeline([('vect', vect), ('nbsvm', classifier)])
    params = {
        'vect__token_pattern': r"\S+",
        'vect__ngram_range': ngram, 
        'vect__binary': True
    }
    clf.set_params(**params)



    print('fitting...')
    clf.fit(train['text'], train['label'])

    print('classifying...')
    pred = clf.predict(test['text'])
   
    print('testing...')
    acc = accuracy_score(test['label'], pred)
    f1 = semeval_senti_f1(pred, test['label'])
    print('NBSVM: acc=%f, f1=%f' % (acc, f1))




advvect = TfidfVectorizer( min_df=0.031, max_df=0.2, max_features = 200000, ngram_range = (2, 2))
advtrain = advvect.fit_transform(theadlines)
print(advtrain.shape)



advancedmodel = NBSVM(C=0.01)
advancedmodel = advancedmodel.fit(advtrain, train["Label"])
teshead = []
for row in range(0,len(test.index)):
    teshead.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advatest = advvect.transform(teshead)
preds13 = advancedmodel.predict(advatest)
acc13 = accuracy_score(test['Label'], preds13)



print('NBSVM: ', acc13)




df = pd.read_csv('data/prices-split-adjusted.csv', index_col=0)
print(df.head())
sel = df[df['symbol'] == 'AAPL']
print(sel.head())
sel.index.sort_values()
sel.index = pd.to_datetime(sel.index, format="%Y/%m/%d")
inp = sel.drop(['symbol','open','low','high','volume'], axis=1)
inp = pd.Series(inp['close'])




def stationarity(ts_data):
    
    rolling_mean = ts_data.rolling(30).mean()
    rolling_std = ts_data.rolling(5).std()

    fig = plt.figure(figsize=(20,10))
    plt.subplot(211)
    plt.plot(ts_data, color='black', label='Original Data')
    plt.plot(rolling_mean, color='red', label='Rolling Mean(30 days)')
    plt.legend()
    plt.subplot(212)
    plt.plot(rolling_std, color='green', label='Rolling Std Dev(5 days)')
    plt.legend()
    
    print('Dickey-Fuller test results\n')
    dicf_test = adfuller(ts_data, regresults=False)
    test_result = pd.Series(dicf_test[0:4], index=['Test Statistic','p-value','# of lags','# of obs'])
    print(test_result)
    for k,v in dicf_test[4].items():
        print('Critical value at %s: %1.5f' %(k,v))



stationarity(inp)




i_log = np.log(inp)
i_log.head()



i_log.dropna(inplace=True)
stationarity(i_log)




i_log_diff = i_log - i_log.shift()




i_log_diff.dropna(inplace=True)
stationarity(i_log_diff)



i_diff = inp - inp.shift()



i_diff.dropna(inplace=True)
stationarity(i_diff)


df_acf = acf(i_diff)




df_pacf = pacf(i_diff)




fig1 = plt.figure(figsize=(20,10))
ax1 = fig1.add_subplot(211)
fig1 = sm.graphics.tsa.plot_acf(df_acf, ax=ax1)
ax2 = fig1.add_subplot(212)
fig1 = sm.graphics.tsa.plot_pacf(df_pacf, ax=ax2)
plt.show()

model = ARIMA(i_diff, (1,1,0))


fit_model = model.fit(full_output=True)


predictions = model.predict(fit_model.params, start=1760, end=1769)



fit_model.summary()


print(predictions)



fit_model.predict(start=1760, end=1769)



pred_diff = pd.Series(fit_model.fittedvalues, copy=True)
print(pred_diff.head())


pred_diff_cumsum = pred_diff.cumsum()
pred_diff_cumsum.head()


inp_trans = inp.add(pred_diff_cumsum, fill_value=0)
inp_trans.tail()

inp.tail()



plt.figure(figsize=(20,10))
plt.plot(inp, color='black', label='Original data')
plt.plot(inp_trans, color='red', label='Fitted Values')
plt.legend()
plt.show()

x = inp.values
y = inp_trans.values



plt.figure(figsize=(20,8))
plt.plot((x - y), color='red', label='Delta')
plt.axhline((x-y).mean(), color='black', label='Delta avg line')
plt.legend()


final_pred = []
for i in predictions:
    t = inp[-1] + i
    final_pred.append(t)



final_pred = pd.Series(final_pred)
final_pred

