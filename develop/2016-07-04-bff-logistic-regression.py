# Code for logistic regression
# Numerai tournament
# Bakeforfun
# 04.07.2016

# INITIALIZATION
import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
from sklearn.metrics import log_loss
# from packages.helpers import plot_learning_curve

# DATA PREPARATION
train = pd.read_csv(os.getcwd() + '/data/numerai_training_data.csv')
test = pd.read_csv(os.getcwd() + '/data/numerai_tournament_data.csv')
example = pd.read_csv(os.getcwd() + '/data/example_predictions.csv')

X = train.drop('target', axis=1)
y = train.target

Xtest = test.drop('t_id', axis=1)
ID = test.t_id

# TRAIN-VALIDATION SPLIT
Xtr, Xval, ytr, yval = cross_validation.train_test_split(X, y, test_size=0.15, random_state=42)


# LOGISTIC REGRESSION
lr = LogisticRegression()

# calculate performance on a validation set
lr.fit(Xtr, ytr)
lr_pred = lr.predict_proba(Xval)
log_loss(yval,lr_pred)

# calculate performance using k-fold cross-validation
CVscores = cross_validation.cross_val_score(lr, X, y, scoring='log_loss', cv=5)

# search for the regularization parameter
Cs = 10**np.linspace(-4, 4, num=15)
grid = {'C': Cs}
gridsearch = grid_search.GridSearchCV(lr, grid, scoring='log_loss', cv=5)
gridsearch.fit(X, y)
gridscores = [-x.mean_validation_score for x in gridsearch.grid_scores_]
# plt.plot(Cs, gridscores)
# plt.scatter(Cs, gridscores)
# plt.scatter(Cs[np.argmin(gridscores)], gridscores[np.argmin(gridscores)], c='g', s=100)
# plt.xscale('log')
# plt.savefig('figures/lr_gridsearch.png')
C = Cs[np.argmin(gridscores)]

# refit the model with the new regularization parameter
lrCV = LogisticRegression(C=C)
lrCV.fit(Xtr, ytr)

# write logloss to file

filename = 'logs/lr_logloss.txt'
dir = os.path.dirname(filename)
if not os.path.exists(dir):
    os.makedirs(dir)
    # os.chmod(dir, mode=0o777)

f = open(os.getcwd() + '/logs/lr_logloss.txt', 'w')
f.write('This is a test\n')
logloss_train = log_loss(ytr, lrCV.predict_proba(Xtr))
logloss_val = log_loss(ytr, lrCV.predict_proba(Xval))
f.write('Train logloss: ' + str(logloss_train) + '\n')
f.write('Validation logloss: ' + str(logloss_val) + '\n')
f.close()

# plot learning curve
# plot_learning_curve(lrCV, "Learning curve", Xtr, ytr, cv=5, train_sizes=np.linspace(0.1, 1, 10), scoring='log_loss')

# refit using all data
lrCV.fit(X, y)

# SUBMISSION
lr_pred = lrCV.predict_proba(Xtest)
lr_submit = pd.DataFrame(lr_pred[:, 1], index=ID, columns={'probability'})

# check if directory exists
filename = 'output/lr_submit.csv'
dir = os.path.dirname(filename)
if not os.path.exists(dir):
    os.makedirs(dir)
    # os.chmod(dir, mode=0o777)

lr_submit.to_csv(os.getcwd() + '/output/lr_submit.csv')