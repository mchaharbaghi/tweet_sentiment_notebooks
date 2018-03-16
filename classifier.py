import pickle
import nltk
import tweet_processing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm


def train_classifier(corpus_file, train_set_number, debug, classifier, labels):
    # See http://www.nltk.org/book/ch06.html
    # http://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost
    # https://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
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

        if filtered_labels:
            # Create feature vectors
            vectorizer = TfidfVectorizer(min_df=1,
                                         max_df=1.0,
                                         sublinear_tf=True,
                                         use_idf=True)
            train_vectors = vectorizer.fit_transform(filtered_tweets)
            # test_vectors = vectorizer.transform(test_data)

            vectorizer = vectorizer

            # Perform classification with SVM
            classifier_linear = svm.LinearSVC()

            classifier_linear.fit(train_vectors, filtered_labels)

            classifier_obj = classifier_linear
            return classifier_obj, vectorizer, train_vectors  # , test_vectors
        else:
            return False, False, False


def classifier_accuracy_bayes(self, training_set_size, corpus_len, accuracy_test_size, corpus, debug):
    if debug:
        print("=====================================")
        print("Calculating classifier accuracy:\n")

    # Overall accumulation variables
    count = 0
    average_accuracy = 0;
    average_tpr = 0
    average_tnr = 0
    # tpr_array = [0] * 10
    # fpr_array = [0] * 10

    # To track total ROC figures
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    while training_set_size + accuracy_test_size < corpus_len and count < 10:
        tweets = corpus[training_set_size:training_set_size + accuracy_test_size]

        # Local figure to calculate TPR and TNR
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for tweet in tweets:
            text = tweet[0]
            actual_sentiment = tweet[1]
            sentiment = self.classifier.classify(self.format_sentence(text))
            if actual_sentiment == 'positive' and sentiment == 'positive':
                TP += 1
            elif actual_sentiment == 'negative' and sentiment == 'negative':
                TN += 1
            elif actual_sentiment == 'positive' and sentiment == 'negative':
                FN += 1
            elif actual_sentiment == 'negative' and sentiment == 'positive':
                FP += 1
        if debug:
            print("Result for run " + str(count + 1) + ":")
            print("[TP, TN, FP, FN]")
            print([TP, TN, FP, FN])

        # Calculate TPR and TNR
        tpr = TP / (TP + FN)
        tnr = TN / (TN + FP)

        # Increment total counters
        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN

        if debug:
            print("Accuracy: " + str(((TP + TN) / accuracy_test_size) * 100) + "%")
            print("Sensitivity (TPR): " + str(tpr))
            print("Sensitivity (TNR): " + str(tnr) + "\n")

        # Increment averages
        average_accuracy += ((TP + TN) / accuracy_test_size) * 100
        average_tpr += tpr
        average_tnr += tnr

        # tpr_array[count] = TP / (TP + FN)
        # fpr_array[count] = FP / (FP + TN)

        # Set up for next run through
        count += 1
        training_set_size += accuracy_test_size

    # Divide by number of runs to get average
    average_accuracy = average_accuracy / 10

    print("Results of accuracy/ROC analysis for classifiers:")
    print("=====================================")
    print("Total ROC analysis results ([TP, TN, FP, FN])")
    print(str([total_TP, total_TN, total_FP, total_FN]) + "\n")
    print("Average accuracy: " + str(round(average_accuracy, 3)) + "%")
    print("Average sensitivity (TPR): " + str(round((average_tpr / 10), 3)))
    print("Average specificity (TNR): " + str(round((average_tnr / 10), 3)))
    print("=====================================\n")

    # Uncomment this if I want to plot a ROC space to compare classifiers
    # x = fpr_array # false_positive_rate
    # y = tpr_array # true_positive_rate
    #
    # # This is the ROC curve
    # plt.plot(x, y)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.title('ROC Analysis of Sentiment Analysis Classifier')
    # plt.show()


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