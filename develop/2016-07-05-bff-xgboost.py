# Code for xgboost
# Numerai tournament
# Bakeforfun
# 10.07.2016

# INITIALIZATION
import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

start_time = time.time()

filename = 'logs/xgb_log.txt'
dir = os.path.dirname(filename)
if not os.path.exists(dir):
    os.makedirs(dir)

f = open(os.getcwd() + '/logs/xgb_log.txt', 'w')

# DATA PREPARATION
train = pd.read_csv(os.getcwd() + '/data/numerai_training_data.csv')
test = pd.read_csv(os.getcwd() + '/data/numerai_tournament_data.csv')
example = pd.read_csv(os.getcwd() + '/data/example_predictions.csv')

X = train.drop('target', axis=1)
y = train.target

Xtest = test.drop('t_id', axis=1)
ID = test.t_id

# XGBOOST
dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(Xtest)

# specify parameters via map
param = {'objective': 'binary:logistic'}
num_round = 50
bst = xgb.train(param, dtrain, num_round)

# make prediction


# write log to file
# logloss_train = log_loss(ytr, lrCV.predict_proba(Xtr))
# logloss_val = log_loss(yval, lrCV.predict_proba(Xval))
# f.write('Train logloss: ' + str(logloss_train) + '\n')
# f.write('Validation logloss: ' + str(logloss_val) + '\n')

# SUBMISSION
xgb_pred = bst.predict(dtest)
xgb_submit = pd.DataFrame(xgb_pred, index=ID, columns={'probability'})

# check if directory exists
filename = 'output/lr_submit.csv'
dir = os.path.dirname(filename)
if not os.path.exists(dir):
    os.makedirs(dir)

xgb_submit.to_csv(os.getcwd() + '/output/xgb_submit.csv')

# write execution time to logs
f.write('\nExecution time:\n')
f.write("--- %s seconds ---" % (time.time() - start_time))
f.close()
