import os
import re
import tweepy
import pandas as pd
import numpy as np
from scipy.special import softmax

# import nltk
# nltk.download("stopwords")

from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

API_TOKEN = os.environ.get("API_TOKEN", "")
QUERY_TITLES = [
    "big", "mouse", "mouth", "miracle", "in", "no", "cell", "mencuri", "raden", "saleh", "ngeri-ngeri", "sedap", "one", "piece", 
    "red", "cyberpunk", "edgerunners", "ivanna", "purple", "heart", "kkn", "penari", "desa", "thor", "thunder", "love"
    "jujutsu", "kaisen", "ngeri", "morbius", "she-hulk", "she", "hulk", "sri", "asih", "pengabdi", "setan"
]

LEXICON_ALAY = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")
STOPWORDS = list(set([
    *stopwords.words("indonesian"),
    *["dengan", "ia","bahwa","oleh", "samaaaaaa"],
    *StopWordRemoverFactory().get_stop_words()
]))

STEMMER = StemmerFactory().create_stemmer()

tokenizer = AutoTokenizer.from_pretrained("AsceticShibs/221026OptimizedModel")
model = AutoModelForSequenceClassification.from_pretrained("AsceticShibs/221026OptimizedModel")


def get_tweets(client, query: str) -> list:
    request = client.search_recent_tweets(
        query = query,
        max_results = 15
    )
    tweets = [tweet.text for tweet in request.data]
    return tweets

def preprocess_tweets(tweets: list):
    cln_tweets     = prc_cleaning(tweets)
    notitle_tweets = prc_rmv_title(cln_tweets)
    norm_tweets    = prc_normalization(notitle_tweets)
    nosw_tweets    = prc_rmv_stopwords(norm_tweets)
    stem_tweets    = prc_stemming(nosw_tweets)

    return stem_tweets

def model_predict(tweets: list):
    BATCH_SIZE = 100
    labels = ["postive", "neutral", "negative"]
    scores_all = np.empty((0,len(labels)))
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(tweets), BATCH_SIZE)):
            end_idx = min(start_idx + BATCH_SIZE, len(tweets))
            encoded_input = tokenizer(tweets[start_idx:end_idx], return_tensors="pt", padding=True, truncation=True).to("cpu")

            output = model(**encoded_input)
            scores = output[0].detach().cpu().numpy()
            scores = softmax(scores, axis=1)
            scores_all = np.concatenate((scores_all, scores), axis=0)

            del encoded_input, output, scores
            torch.cuda.empty_cache()
    
    return labels, scores_all

# Preprocess
def prc_cleaning(tweets: list) -> list:
    cln_tweets = []
    for tweet in tweets:
        cln_tweet = tweet.lower()
        cln_tweet = re.sub(r"(?:\@|#|https?\://)\S+", " ", cln_tweet) # eliminate username, url, hashtags
        cln_tweet = re.sub(r'&amp;', '', cln_tweet) # remove &amp; as it equals &
        cln_tweet = re.sub(r'[^\w\s]',' ', cln_tweet) # remove punctuation
        cln_tweet = re.sub(r'[0-9]', '', cln_tweet) # remove number
        cln_tweet = re.sub(r'[\s\n\t\r]+', ' ', cln_tweet) # remove extra space
        cln_tweet = cln_tweet.strip() # trim
        
        cln_tweets.append(cln_tweet)
    return cln_tweets


def prc_rmv_title(tweets: list) -> list:
    notitle_tweets = []
    for tweet in tweets:
        cln_tokens = [
            word 
            for word in tweet.split(" ")
            if (word not in QUERY_TITLES) and (word != "")
        ]

        notitle_tweets.append(" ".join(cln_tokens))
    return notitle_tweets


def prc_normalization(tweets: list) -> list:
    norm_tweets = []
    for tweet in tweets:
        fw = [w for w in tweet.split(" ")]
        example_rec = ""
        for word in fw:
            if (word in LEXICON_ALAY["slang"].to_list()):
                propper = LEXICON_ALAY[LEXICON_ALAY["slang"] == word]["formal"].drop_duplicates().iloc[0]
                example_rec += propper + " "
            elif (word not in LEXICON_ALAY["slang"].to_list()):
                propper = word
                example_rec += propper + " "
        example_rec = example_rec[:-1]
        norm_tweets.append(example_rec)
    return norm_tweets


def prc_rmv_stopwords(tweets: list) -> list:
    nosw_tweets = []
    for tweet in tweets:
        cln_tokens = [word for word in tweet.split(" ") if (word not in STOPWORDS) and (word != "")]
        nosw_tweets.append(" ".join(cln_tokens))
    return nosw_tweets


def prc_stemming(tweets: list) -> list:
    stem_tweets = []
    for tweet in tweets:
        stem_tweet = STEMMER.stem(tweet)
        stem_tweets.append(stem_tweet)
    return stem_tweets


# Main
def predict_sentiment(query: str):
    client = tweepy.Client(bearer_token=API_TOKEN)
    tweets = get_tweets(client, query)
    labels, predictions = model_predict(tweets)
    return labels, predictions

def count_sentiment(labels: list, data: list):
    sentiment_cnt = {}
    for label in labels:
        sentiment_cnt[label] = 0
    
    for row in data:
        sentiment = labels[np.argmax(row)]
        sentiment_cnt[sentiment] += 1
    
    return sentiment_cnt