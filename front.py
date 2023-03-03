from flask import Flask, render_template, request
import pandas as pd
import sklearn
import itertools
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__,template_folder='./templates',static_folder='./static')

loaded_model = pickle.load(open('model.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))
tfidf_v = TfidfVectorizer()
corpus = []