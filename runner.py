# Import Libraries

from typing import final
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import nltk
from tqdm import tqdm
import dataframe_image as dfi
import webbrowser as wb


# Comment below after first run 
# nltk.downloader.download('stopwords')
# nltk.downloader.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import snscrape.modules.twitter as sntwitter
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer

plt.style.use('fivethirtyeight')

tok = WordPunctTokenizer()

# Global Variables

query = input("Enter Keyword : ")
limit = int(input("Enter Tweet Count Limit : "))
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []
pbar = tqdm(total=limit)


# Format Specific Variables

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not", "i'm":"i am"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')


# Tweet text cleaning and formating function

def tweet_formatter(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    updatedText = (" ".join(words)).strip()
    tweet_list.append(updatedText)
    sentimental_analysis(updatedText)


# Sentimental analysing and scoring function

def sentimental_analysis(text):
    analysis = TextBlob(text)
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    global polarity
    global negative
    global positive
    global neutral
    polarity += analysis.sentiment.polarity

    if neg > pos:
        negative_list.append(text)
        negative += 1

    elif pos > neg:
        positive_list.append(text)
        positive += 1

    elif pos == neg:
        neutral_list.append(text)
        neutral += 1

def percentage(part,whole):
 return 100 * float(part)/float(whole)


# Tweet data Scraping

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    if len(tweet_list) == limit:
        pbar.close()
        print("DATA FETCHING SUCCESSFULLY COMPLETED")
        break
    else:
        pbar.update(1)
        # tweet_list.append(tweet.content)
        tweet_formatter(tweet.content)


positive = percentage(positive, limit)
negative = percentage(negative, limit)
neutral = percentage(neutral, limit)
polarity = percentage(polarity, limit)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')


#Number of Tweets (Total, Positive, Negative, Neutral)

tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
tweetDetail = []
text = "total Tweet Number: ",len(tweet_list)
tweetDetail.append(text)
text = "positive Tweet Number: ",len(positive_list)
tweetDetail.append(text)
text = "negative Tweet Number: ", len(negative_list)
tweetDetail.append(text)
text = "neutral Tweet Number: ",len(neutral_list)
tweetDetail.append(text)

tweetDetail = pd.DataFrame(tweetDetail)
dfi.export(tweetDetail, 'results/tweetDetail.png')


# Saving Dataset

tweet_list.to_csv('results/dataset.csv')
print("dataset Saved Successfully")


#Creating PieCart

labels = ['Positive ['+str(positive)+'%]' , 'Neutral ['+str(neutral)+'%]','Negative ['+str(negative)+'%]']
sizes = [positive, neutral, negative]
colors = ['green', 'blue','red']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment Analysis on keyword : " + query)
plt.axis('equal')
plt.savefig('results/PieCart.png')
print("Piechart Saved Successfully")

# Eliminating duplicates and creating new DF

tweet_list.drop_duplicates(inplace = True)
tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]


#Removing RT, Punctuation etc

remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()


#Calculating Negative, Positive, Neutral and Compound values

tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
        tw_list.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
        tw_list.loc[index, 'sentiment'] = "positive"
    else:
        tw_list.loc[index, 'sentiment'] = "neutral"
    tw_list.loc[index, 'neg'] = neg
    tw_list.loc[index, 'neu'] = neu
    tw_list.loc[index, 'pos'] = pos
    tw_list.loc[index, 'compound'] = comp


#Creating new data frames for all sentiments (positive, negative and neutral)

tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]


#Function for count_values_in single columns

def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])


#Count_values for sentiment

count_values_in_column(tw_list,"sentiment")


#Function to Create Wordcloud

def create_wordcloud(text, name):
    mask = np.array(Image.open("cloud.png"))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
                  mask = mask,
                  max_words=3000,
                  stopwords=stopwords,
                  repeat=True)
    wc.generate(str(text))
    wc.to_file("results/" + name + "_wc.png")
    print(name + " Word Cloud Saved Successfully")


#Creating wordcloud for positive sentiment

create_wordcloud(tw_list_positive["text"].values, "Positive")


#Creating wordcloud for negative sentiment

create_wordcloud(tw_list_negative["text"].values, "Negative")


#Creating wordcloud for neutral sentiment

create_wordcloud(tw_list_neutral["text"].values, "Neutral")


#Calculating tweet's lenght and word count

tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))
round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()),2)
round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()),2)


#Removing Punctuation

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))


#Appliyng tokenization

def tokenization(text):
    text = re.split('\W+', text)
    return text

tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))


#Removing stopwords

stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))


#Appliyng Stemmer

ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))


#Cleaning Text

def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

#Appliyng Countvectorizer

countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(tw_list['text'])
count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names_out()) # get_feature_names is getting debricated in 1.2

# Most Used Words

count = pd.DataFrame(count_vect_df.sum())
countdf = count.sort_values(0,ascending=False).head(20)
countdf[1:11]
Mostcount = countdf[1:11]
dfi.export(Mostcount, 'results/MostUsedWords.png')


# Ngram model helps us to predict most probabl2 word that might follow the sequence
#Function to ngram

def get_top_n_gram(corpus,ngram_range,n=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


#n2_bigram

n2_bigrams = get_top_n_gram(tw_list['text'],(2,2),20)
n2_bigrams = pd.DataFrame(n2_bigrams)
dfi.export(n2_bigrams, 'results/n2_bigrams.png')

wb.register('chrome', None)
wb.open("results/index.html")