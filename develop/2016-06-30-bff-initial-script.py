# Initial script for Numerai tournament
# Bakeforfun
# 30.06.2016

# INITIALIZATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.metrics import log_loss
from packages.helpers import plot_learning_curve

# DATA PREPARATION
train = pd.read_csv("data/numerai_training_data.csv")
test = pd.read_csv("data/numerai_tournament_data.csv")
example = pd.read_csv("data/example_predictions.csv")

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
plt.plot(Cs, gridscores)
plt.scatter(Cs, gridscores)
plt.scatter(Cs[np.argmin(gridscores)], gridscores[np.argmin(gridscores)], c='g', s=100)
plt.xscale('log')
C = Cs[np.argmin(gridscores)]

# refit the model with the new regularization parameter
lrCV = LogisticRegression(C=C)
lrCV.fit(Xtr, ytr)
print('Train logloss', log_loss(ytr, lrCV.predict_proba(Xtr)))
print('Validation logloss', log_loss(yval, lrCV.predict_proba(Xval)))

plot_learning_curve(lrCV, "Learning curve", Xtr, ytr, cv=5, train_sizes=np.linspace(0.1, 1, 10), scoring='log_loss')

lrCV.fit(X, y)

# GAUSSIAN NAIVE BAYES
gnb = GaussianNB()
gnb.fit(Xtr, ytr)
print('Train logloss', log_loss(ytr, gnb.predict_proba(Xtr)))
print('Validation logloss', log_loss(yval, gnb.predict_proba(Xval)))


# RANDOM FORESTS
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
rf.fit(Xtr, ytr)
print('Train logloss', log_loss(ytr, rf.predict_proba(Xtr)))
print('Validation logloss', log_loss(yval, rf.predict_proba(Xval)))

calib = CalibratedClassifierCV(RandomForestClassifier(n_jobs=-1, n_estimators=200), cv=5)
calib.fit(Xtr, ytr)
print('Train logloss', log_loss(ytr, calib.predict_proba(Xtr)))
print('Validation logloss', log_loss(yval, calib.predict_proba(Xval)))

calib = CalibratedClassifierCV(RandomForestClassifier(n_jobs=-1, n_estimators=200), cv=5, method='isotonic')
calib.fit(Xtr, ytr)
print('Train logloss', log_loss(ytr, calib.predict_proba(Xtr)))
print('Validation logloss', log_loss(yval, calib.predict_proba(Xval)))

# SVM
svm = SVC(probability=True)
svm.fit(Xtr, ytr)
print('Train logloss', log_loss(ytr, svm.predict_proba(Xtr)))
print('Validation logloss', log_loss(yval, svm.predict_proba(Xval)))

# SUBMISSION
lr_pred = lrCV.predict_proba(Xtest)
lr_submit = pd.DataFrame(lr_pred[:, 1], index=ID, columns={'probability'})
lr_submit.to_csv('output/logResSubmit.csv')
