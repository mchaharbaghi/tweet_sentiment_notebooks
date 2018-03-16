from __future__ import print_function

import nltk
import sys
import os

# My imports =====
import corpus
import tweet_processing
import classifier
# ================

from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


class SentimentAnalysis:
    def __init__(self, testing, api, classifier, corpus_num, tweets_num):
        self.api = api

        self.testing = testing

        self.word_features = []
        self.classifier = None

        self.bool_run_setup = False

        self.classifier_type = classifier  # Options: bayes, svm

        self.bayes_corpus_num = int(corpus_num)  # Options: 1, 2, 3
        self.svm_corpus_num = int(corpus_num)  # Options: 1, 2, 3

        self.tweets_num = int(tweets_num)

        self.bayes_training_set_size = 100000
        self.svm_training_set_size = 10000

        self.vectorizer = None


    # Run-time methods =======================================================
    def analyse_tweet(self, tweet, method):
        if method == "bayes":
            predict = self.classifier.classify(tweet_processing.format_sentence(tweet))
            dist = self.classifier.prob_classify(tweet_processing.format_sentence(tweet))

            for label in dist.samples():
                if label == 'positive':
                    positive = dist.prob(label)
                elif label == 'negative':
                    negative = dist.prob(label)

            return predict, positive, negative
        elif method == "svm":
            test_vector = self.vectorizer.transform(tweet)
            prediction = self.classifier.predict(test_vector)
            probs = self.classifier.decision_function(test_vector)
            negative = (1 - probs) / 2
            positive = 1 - negative
            return prediction[0], positive, negative

    def run_bayes(self):
        # self.training_set_size = min(len(corpus), self.bayes_training_set_size)
        my_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "classifiers")
        classifier_file = os.path.join(my_path, "bayes_corpus_" + str(self.bayes_corpus_num) + ".pickle")
        if os.path.isfile(classifier_file):
            return classifier.open_classifier(classifier_file, False)

    def run_svm(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, "corpus_" + str(self.svm_corpus_num) + "/corpus_analysis.csv")
        if os.path.isfile(path):
            corpus_file, labels = corpus.import_corpus_load(path, False, "svm")
        else:
            corpus_file, labels, test_data, test_labels = corpus.import_corpus_setup(False, self.svm_corpus_num, self.classifier_type)

        self.training_set_size = min(len(corpus_file),  self.svm_training_set_size)
        corpus_file = corpus_file[:int(self.training_set_size)]
        labels = labels[:int(self.training_set_size)]

        my_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "classifiers")
        classifier_file = os.path.join(my_path, "svm_corpus_" + str(self.svm_corpus_num) + ".pickle")
        if os.path.isfile(classifier_file) and not self.bool_run_setup:
            classifier_obj = classifier.open_classifier(classifier_file, False)
            vectorizer = TfidfVectorizer(min_df=5,
                                         max_df=0.8,
                                         sublinear_tf=True,
                                         use_idf=True)
            train_vectors = vectorizer.fit_transform(corpus_file)
            #test_vectors = vectorizer.transform(test_data)

            classifier_obj.fit(train_vectors, labels)

            return classifier_obj, vectorizer
        else:
            classifier_obj, vectorizer, train_vectors = classifier.train_classifier(corpus_file, self.training_set_size, False, self.classifier_type, labels)  # TODO: Contrast different automated ways to train Naive Bayesian Classifier
            classifier.save_classifier(str(classifier_file), False)
            classifier_obj.fit(train_vectors, labels)
            return classifier, vectorizer
    # =========================================================================


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)