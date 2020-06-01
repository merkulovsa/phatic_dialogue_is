# -*- coding: utf-8 -*-
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import os
import pymorphy2
import re
import chardet

import nltk

nltk.download('popular', quiet=True) # for downloading packages

# uncomment the following only the first time
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only
morph = pymorphy2.MorphAnalyzer()

def Lemmatizer(text):
    # Detecting codepage, recoding, cleaning, lemmatizing
    codepage = chardet.detect(text.encode())['encoding']
    text = (text.encode()).decode(codepage)
    text = " ".join(word.lower() for word in text.split())  # lowercasing and removing short words
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
    text = re.sub('[.,:;%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
    text = " ".join(morph.parse(str(word))[0].normal_form for word in text.split())
    text = text.encode("utf-8")
    return nltk.word_tokenize(text.decode())

with open('chatbot.txt', 'r', encoding='utf-8', errors='ignore') as fin:
    raw = fin.read()
with open('stopwords.txt', 'r', encoding='utf-8', errors='ignore') as fin:
    stopwords = fin.read()
#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words
stop_words = nltk.word_tokenize(stopwords)

# Keyword Matching
GREETING_INPUTS = ("здравствуй", "добрый день", "доброе утро", "добрый вечер", "доброго времени суток", "приветствую", "привет", "мое почтение", "приветик", "хей", "здарова", "салют", "Сколько лет, сколько зим!", "рад видеть", "ку")
GREETING_RESPONSES = ["Здравствуй", "Доброго времени суток", "Приветствую", "Мое почтение", "Сколько лет, сколько зим!", "Рад видеть"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=Lemmatizer, stop_words=stop_words)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        robo_response = robo_response + "Хмм...Я пока не готов ответить на этот вопрос, нужно еще поразмыслить. Может сменим тему?"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


flag = True
print("Приветствую. Я философ-бот и постараюсь поддержать ваш разговор.")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'пока'):
        if(user_response == 'спасибо'):
            flag = False
            print("Приходи еще!")
        else:
            if(greeting(user_response) != None):
                print(greeting(user_response))
            else:
                print(end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("Бывай! Береги себя..")