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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

directory = os.getcwd()
short_pos = open(f'{directory}/Train_and_Test_data/Positive_data.txt','r').read()
short_neg = open(f'{directory}/Train_and_Test_data/Negative_data.txt','r').read()
documents = []


all_words = []


lemmatizer = WordNetLemmatizer()

for r in short_pos.split('\n'):
    documents.append((r,'neg'))
    words = word_tokenize(r)
    for w in words:
        if w not in stopwords.words():
            all_words.append(lemmatizer.lemmatize(w))
        
for r in short_neg.split('\n'):
    documents.append((r,'pos'))
    words = word_tokenize(r)
    for w in words:
        if w not in stopwords.words():
            all_words.append(lemmatizer.lemmatize(w))



save_documents = open(f'{directory}/pickled_algos/documents.pickle','wb')
pickle.dump(documents,save_documents)
save_documents.close()
    

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:8000]

save_word_features = open(f'{directory}/pickled_algos/word_features.pickle','wb')
pickle.dump(word_features,save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featuresets = [(find_features(rev), catagory)
               for (rev, catagory) in documents]
random.shuffle(featuresets)

save_featuresets = open(f'{directory}/pickled_algos/featuresets.pickle','wb')
pickle.dump(featuresets,save_featuresets)
save_featuresets.close()


training_set = featuresets[:24000]
testing_set = featuresets[24000:]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Baise Algo accuracy:', (nltk.classify.accuracy(classifier, testing_set)))


save_classifier =open(f'{directory}/pickled_algos/naivebayes.pickle','wb')
pickle.dump(classifier,save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('MNB_classifier accuracy:', (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_MNB_classifier = open(f'{directory}/pickled_algos/MNB_classifier.pickle','wb')
pickle.dump(MNB_classifier,save_MNB_classifier)
save_MNB_classifier.close()


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print('BernoulliNB_classifier accuracy:', (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_BernoulliNB_classifier = open(f'{directory}/pickled_algos/BernoulliNB_classifier.pickle','wb')
pickle.dump(BernoulliNB_classifier,save_BernoulliNB_classifier)
save_BernoulliNB_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print('LogisticRegression_classifier accuracy:', (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_LogisticRegression_classifier = open(f'{directory}/pickled_algos/LogisticRegression.pickle','wb')
pickle.dump(LogisticRegression_classifier,save_LogisticRegression_classifier)
save_LogisticRegression_classifier.close()

SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print('SGD_classifier accuracy:', (nltk.classify.accuracy(SGD_classifier, testing_set))*100)

save_SGD_classifier = open(f'{directory}/pickled_algos/SGD_classifier.pickle','wb')
pickle.dump(SGD_classifier,save_SGD_classifier)
save_SGD_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC_classifier accuracy:', (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_LinearSVC_classifier = open(f'{directory}/pickled_algos/LinearSVC_classifier.pickle','wb')
pickle.dump(LinearSVC_classifier,save_LinearSVC_classifier)
save_LinearSVC_classifier.close()


Voted_classifier = VoteClassifier(classifier, LinearSVC_classifier, SGD_classifier, LogisticRegression_classifier, MNB_classifier, BernoulliNB_classifier)

print('voted_classifier confidence:', (nltk.classify.accuracy(Voted_classifier, testing_set))*100,Voted_classifier.classify(testing_set[0][0]))

def sentiment(text):
    feats = find_features(text)
    return Voted_classifier.classify(feats)


















































