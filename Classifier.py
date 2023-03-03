import DataPrep
import FeatureSelection
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# string to test
doc_new = ['obama is running for president in 2016']

# the feature selection has been done in FeatureSelection.py module. here we will create models using those features
# for prediction

# first we will use bag of words techniques

# building classifier using naive bayes
nb_pipeline = Pipeline([
    ('NBCV', FeatureSelection.countV),
    ('nb_clf', MultinomialNB())])

nb_pipeline.fit(DataPrep.train_news['Statement'], DataPrep.train_news['Label'])
predicted_nb = nb_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_nb == DataPrep.test_news['Label'])

# building classifier using logistic regression
logR_pipeline = Pipeline([
    ('LogRCV', FeatureSelection.countV),
    ('LogR_clf', LogisticRegression())
])

logR_pipeline.fit(DataPrep.train_news['Statement'], DataPrep.train_news['Label'])
predicted_LogR = logR_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_LogR == DataPrep.test_news['Label'])
