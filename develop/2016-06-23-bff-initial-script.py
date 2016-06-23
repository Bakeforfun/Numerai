# Initial script for Numerai tournament
# Bakeforfun
# 23.06.2016

# Initialisation
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Data preparation
train = pd.read_csv("data/numerai_training_data.csv")
test = pd.read_csv("data/numerai_tournament_data.csv")
example = pd.read_csv("data/example_predictions.csv")

X = train.ix[:, 0:21]
y = train.ix[:, 21]

# Logistic regression
logRes = LogisticRegression()
logRes.fit(X, y)
logResPred = logRes.predict_proba(X)

# Submit