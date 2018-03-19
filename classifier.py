import pickle
import nltk
import tweet_processing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm


def train_classifier(corpus_file, train_set_number, debug, classifier, labels):
    if debug:
        print("Training classifier: " + classifier + "\n")

    if int(train_set_number) > 0:
        tweets = corpus_file[:int(train_set_number)]
        labels = labels[:int(train_set_number)]

    if classifier == 'bayes':
        filtered_tweets = []
        count = 0
        for entry in tweets:
            text = entry[0]
            sentiment = entry[1]
            if sentiment == 'positive' or sentiment == 'negative' or sentiment == 'neutral':
                entry = [tweet_processing.format_sentence(text), sentiment]
                filtered_tweets.append(entry)
                count += 1
        if filtered_tweets:
            classifier_obj = nltk.NaiveBayesClassifier.train(filtered_tweets)
            return classifier_obj
        else:
            return False

    elif classifier == 'svm':
        count = 0
        filtered_tweets = []
        filtered_labels = []
        for label in labels:
            if label == "positive" or label == "negative" or label == "neutral":
                filtered_tweets.append(tweets[count])
                filtered_labels.append(label)
                count += 1

        if not filtered_labels:
            return False, False, False

        # Create feature vectors
        vectorizer = TfidfVectorizer(min_df=1,
                                     max_df=0.95,
                                     sublinear_tf = True,
                                     use_idf = True,
                                     ngram_range=(1, 2)
                                     )
        train_vectors = vectorizer.fit_transform(filtered_tweets)

        vectorizer = vectorizer

        # Perform classification with SVM
        classifier_linear = svm.LinearSVC(C=0.1)
        classifier_linear.fit(train_vectors, filtered_labels)

        classifier_obj = classifier_linear
        return classifier_obj, vectorizer, train_vectors  # , test_vectors


def save_classifier(classifier, path, debug):
    if debug:
        print("Saving classifier to file\n")
    f = open(path, 'wb')
    pickle.dump(classifier, f)
    f.close()


def open_classifier(path, debug):
    if debug:
        print("Opening classifier from file\n")
    f = open(path, 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier
