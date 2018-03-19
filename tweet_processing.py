import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer

def preprocess_tweets(tweets, debug):
    # Delete tweets which return "Not Available"
    available_tweets = [x for x in tweets if x != "Not Available!"]

    processed_tweets = []

    for tweet in available_tweets:
        if type(tweet) is not str:
            tweet = tweet.text
        if tweet != None:
            # Strip links
            links = bool(re.search(r"\\https|http.?:\/\/[^\s]+[\s]?", tweet))
            new_tweet = re.sub(r"\\https|http.?:\/\/[^\s]+[\s]?", '', tweet)

            # Strip RT from tweet
            retweet = bool(re.search('RT\s*', new_tweet))
            new_tweet = re.sub(r'RT\s*', '', new_tweet)

            # Replace encoded ampersands
            new_tweet = re.sub(r'&amp;', 'and ', new_tweet)

            # Replace #word with word
            new_tweet = re.sub(r'#([^\s]+)', r'\1', new_tweet)

            # Repeating words like happyyyyyyyy
            #repeating = re.compile(r"(.)\1{1,}", re.IGNORECASE)
            new_tweet = re.sub(r"(.)\1+", r"\1\1", new_tweet)

            # Strip any numbers
            # number = bool(re.search(r"\s?[0-9]+\.?[0-9]*", new_tweet))
            # new_tweet = re.sub(r"\s?[0-9]+\.?[0-9]*", '', new_tweet)

            # Strip special characters
            spec_char = bool(re.search(r"\?|\!|\#|\(|\)|\"|\.|\,|\*", new_tweet))
            new_tweet = re.sub("\?|\!|\#|\(|\)|\"|\.|\,|\*", " ", new_tweet)

            # Strip any mentions
            mentions = bool(re.search('@\w+:*\s{0,1}', new_tweet))
            new_tweet = re.sub(r'@\w+:*\s{0,1}', '', new_tweet)

            # Strip emojis
            # Source: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
            old_tweet = new_tweet
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001FE0F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       "]+", flags=re.UNICODE)
            new_tweet = emoji_pattern.sub(r'', new_tweet)

            # Remove text emoticons e.g. :) - Regex taken from https://stackoverflow.com/questions/14571103/capturing-emoticons-using-regular-expression-in-python
            new_tweet = re.sub(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)", '', new_tweet)
            emoji = old_tweet != new_tweet

            # Strip new lines
            new_tweet = re.sub(r'/\r?\n|\r/g', ' ', new_tweet)

            final_tweet = ""

            stop_words = set(stopwords.words('english'))
            tweet_tokenize = TweetTokenizer()

            for word in tweet_tokenize.tokenize(new_tweet):
                if word == "":
                    continue
                if (len(word) >= 3) and word not in stop_words:
                    final_tweet = final_tweet + word + " "

            # Convert tweets to lowercase finally
            final_tweet = final_tweet.lower()

            # Ensure we don't just have an empty tweet now before saving it, if we do just discard it as it's no use
            if re.search('[a-zA-Z]', final_tweet):
                processed_tweets.append(final_tweet)
            else:
                processed_tweets.append("NULL")

    processed_tweets = [x.rstrip() for x in processed_tweets]
    return processed_tweets


def get_words_in_tweets(tweets):  # https://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def format_sentence(sent):  # https://www.twilio.com/blog/2017/09/sentiment-analysis-python-messy-data-nltk.html
    tokenizer = RegexpTokenizer(r'\w+')
    return {word: True for word in tokenizer.tokenize(sent)}
