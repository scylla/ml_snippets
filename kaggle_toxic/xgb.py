from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle

TRAIN_DATA_FILE = "/Users/amitn/Documents/kaggle/toxicity/train.csv"
TEST_DATA_FILE = "/Users/amitn/Documents/kaggle/toxicity/test.csv"
MAX_FEATURES = 20000

train = pd.read_csv(TRAIN_DATA_FILE)
list_sentences_train = train["comment_text"]
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values

tfidf_vectorizer_desc = TfidfVectorizer(lowercase=True, max_features=MAX_FEATURES)
x = tfidf_vectorizer_desc.fit_transform(list_sentences_train)

clf_multilabel = OneVsRestClassifier(XGBClassifier())
clf_multilabel.fit(x,y)

test = pd.read_csv(TEST_DATA_FILE)
list_sentences_test = test["comment_text"]
x_test = tfidf_vectorizer_desc.transform(list_sentences_test)
y_test = clf_multilabel.predict_proba(x_test)

sample_submission = pd.read_csv("/Users/amitn/Documents/kaggle/toxicity/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("/Users/amitn/Documents/kaggle/toxicity/baseline.csv", index=False)
