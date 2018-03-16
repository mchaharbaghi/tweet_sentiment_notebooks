import csv
import os
import sys
import nltk
nltk.download('twitter_samples', quiet=True)
from nltk.corpus import twitter_samples

def import_corpus_setup(debug, corpus, classifier):
    if debug:
        print("Importing corpus\n")
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    if corpus == 1:
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "corpus_1/corpus.csv"), 'r') as f:
            reader = csv.reader(f)
            final_list = []
            for row in reader:
                if classifier == 'bayes':
                    new_row = [row[2]]
                    # print(row[1])
                    if row[1] == "1":
                        new_row.append("positive")
                    elif row[1] == "0":
                        new_row.append("negative")
                    final_list.append(new_row)
                elif classifier == 'svm':
                    train_data.append(row[0])
                    if row[1] == "1":
                        train_labels.append("positive")
                    elif row[1] == "0":
                        train_labels.append("negative")
        if classifier == 'bayes':
            # if debug:
            #     for rows in final_list:
            #         print(rows)
            return final_list
        elif classifier == 'svm':
            return train_data, train_labels, test_data, test_labels
        else:
            eprint('Invalid classifier option')
            exit(1)
    elif corpus == 2:
            positiveTweets = twitter_samples.strings('positive_tweets.json')
            negativeTweets = twitter_samples.strings('negative_tweets.json')

            positiveTweets = [(x, "positive") for x in positiveTweets]
            negativeTweets = [(x, "negative") for x in negativeTweets]
            tweets = positiveTweets + negativeTweets

            if classifier == "bayes":
                return tweets

            elif classifier == "svm":
                tweet_label = []
                svm_tweets = [x[0] for x in tweets]
                for tweet in tweets:
                    if tweet[1] == "positive":
                        tweet_label.append("positive")
                    elif tweet[1] == "negative":
                        tweet_label.append("negative")
                return svm_tweets, tweet_label, test_data, test_labels
    elif corpus == 3:
        data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "corpus_3")
        classes = ['positive', 'negative']

        # Read the data
        for curr_class in classes:
            dirname = os.path.join(data_dir, curr_class)
            for fname in os.listdir(dirname):
                with open(os.path.join(dirname, fname), 'r') as f:
                    content = f.read()
                    if classifier == 'svm' and fname.startswith('cv9'):
                        test_data.append(content)
                        test_labels.append(curr_class)
                    else:
                        if classifier == 'bayes':
                            train_data_row = [content]
                            if curr_class == 'positive':
                                train_data_row.append("positive")
                            else:
                                train_data_row.append("negative")
                            train_data.append(train_data_row)
                        else:
                            train_data.append(content)
                            train_labels.append(curr_class)
        if classifier == "svm":
            return train_data, train_labels, test_data, test_labels
        elif classifier == "bayes":
            return train_data


def import_corpus_load(corpus_file, debug, classifier_type):
    train_data = []
    train_labels = []
    with open(corpus_file, 'r') as f:
        reader = csv.reader(f)
        if classifier_type == "bayes":
            your_list = list(reader)
            return your_list
        elif classifier_type == "svm":
            for row in reader:
                train_data.append(row[0])
                if row[1] == "positive":
                    train_labels.append('positive')
                elif row[1] == "negative":
                    train_labels.append('negative')
            return train_data, train_labels


def save_corpus(self, corpus_list, debug, path):
    list = corpus_list
    if debug:
        print("Saving corpus CSV")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerows(list)

def print_corpus_lines(corpus_file):
    with open(corpus_file, 'r') as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            count += 1
            if count > 5:
                break
            print("Tweet: " + row[0])
            if row[1] == "positive":
                print("Sentiment: Positive")
            elif row[1] == "negative":
                print("Sentiment: Negative")

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)