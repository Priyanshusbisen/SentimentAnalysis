import pickle
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self,*Classifiers):
        self._classifiers = Classifiers

    def classify(self,features):
        votes = []
        for c in self._classifiers:
            votes.append(c.classify(features))
            
            return mode(votes)


    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            votes.append(c.classify(features))

        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf


open_file = open('pickled_algos/documents.pickle','rb')
documents = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/word_features.pickle','rb')
word_features = pickle.load(open_file)
open_file.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

open_file = open('pickled_algos/MNB_classifier.pickle','rb')
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/naivebayes.pickle','rb')
NaiveBayes_classifier = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/LinearSVC_classifier.pickle','rb')
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/LogisticRegression.pickle','rb')
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/SGD_classifier.pickle','rb')
SGD_classifier = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/BernoulliNB_classifier.pickle','rb')
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

#open_file = open('pickled_algos/NuSVC_classifier.pickle','rb')
#NuSVC_classifier = pickle.load(open_file)
#open_file.close()


Voted_Classifier = VoteClassifier(SGD_classifier,MNB_classifier,NaiveBayes_classifier,LinearSVC_classifier,LogisticRegression_classifier,BernoulliNB_classifier)

def sentiment(text):
    feat = find_features(text)
    if Voted_Classifier.confidence(feat)*100 >= 65:
        return Voted_Classifier.classify(feat)
    else:
        return None












