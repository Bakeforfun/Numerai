# Initial script for Numerai tournament
# Bakeforfun
# 23.06.2016

# INITIALIZATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.metrics import log_loss

# DATA PREPARATION
train = pd.read_csv("data/numerai_training_data.csv")
test = pd.read_csv("data/numerai_tournament_data.csv")
example = pd.read_csv("data/example_predictions.csv")

X = train.ix[:, 0:21]
y = train.ix[:, 21]

Xtest = test.ix[:, 1:22]
ID = test.ix[:, 0]

# LOGISTIC REGRESSION
logRes = LogisticRegression()

# Split the data into training and test set for evaluation
Xtr, Xval, ytr, yval = cross_validation.train_test_split(X, y, test_size=0.15, random_state=42)
# calculate performance on a validation set
logRes.fit(Xtr, ytr)
logResPred = logRes.predict_proba(Xval)
log_loss(yval,logResPred)

# calculate performance using k-fold cross-validation
CVscores = cross_validation.cross_val_score(logRes, X, y, scoring='log_loss', cv=5)

# search for the regularization parameter
Cs = 10**np.linspace(-4, 4, num=15)
grid = {'C': Cs}
gridsearch = grid_search.GridSearchCV(logRes, grid, scoring='log_loss', cv=5)
gridsearch.fit(X, y)
gridscores = [-x.mean_validation_score for x in gridsearch.grid_scores_]
plt.plot(Cs, gridscores)
plt.scatter(Cs, gridscores)
plt.scatter(Cs[np.argmin(gridscores)], gridscores[np.argmin(gridscores)], c='g', s=100)
plt.xscale('log')
C = Cs[np.argmin(gridscores)]

# refit the model with the new regularization parameter
logResCV = LogisticRegression(C=C)
logResCV.fit(Xtr, ytr)
print('Train logloss', log_loss(ytr, logResCV.predict_proba(Xtr)))
print('Validation logloss', log_loss(yval, logResCV.predict_proba(Xval)))
logResCV.fit(X, y)

# SUBMISSION
logResPred = logRes.predict_proba(Xtest)
logResSubmit = pd.DataFrame(logResPred[:, 1], index=ID, columns={'probability'})
logResSubmit.to_csv('output/logResSubmit.csv')
