import numpy as np
import pandas as pd
import datetime
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import json
import sys
import os
import pandas as pd
import numpy as np
import re
import pyvi
from pyvi import ViTokenizer
import pickle
from gensim.models import Word2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from libs.utils import *
FORMAT_DATE = r"%d/%m/%Y %H:%M:%S"

feature_vocabulary = {
    "abc" : "Positive",
    'def' : "Negative",
    "ghi" : "Neutral",
 }
def _max(pos, neg, neu):
    if (pos >= neg and pos >= neu):
        return "Positive"
    if (neg >= pos and neg >= neu):
        return "Negative"
    return "Neutral"

def _sum(lst):
    res = 0
    for i in lst:
        res += i

    
    return res

def Predict(text):
    if text == None or text == "":
        return None
    # file repair word
    file = open(os.path.join(os.getcwd(),"repairword.txt"), encoding="utf-8")
    repair_words_list = dict(line.strip().split(":") for line in file)

    # Load model Word2Vec
    model = Word2Vec.load(os.path.join(os.getcwd(),"model_vocal_sw.bin"))

    # load module logicstic regression
    with open(os.path.join(os.getcwd(),"logistic_regression_Vocal_SW.pkl"), 'rb') as f:
        models = pickle.load(f)

    # load file vocalbulary
    feature_vocabulary = load_model(os.path.join(os.getcwd(),"feature_vocabulary.pkl"))

    # replace endline
    text = text.replace("\n", " ")
    
    # remove link, gmail, tags, hastags, phonenumber
    text = re.sub(r'https\S+', " ", text)
    text = re.sub(r'http\S+', " ", text)
    text = re.sub(r'@\w+(\.com(\.vn))?', " ", text)
    text = re.sub(r'#\w+', " ", text)
    text = re.sub(r'\d{10}', " ", text)

    # remove special character
    text = re.sub(r'[^a-zA-Z0-9\sÁ-ỹ]', " ", text)

    # remove emoji
    emoji = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        u"\U000023F0"
                      "]+", re.UNICODE)
    text = re.sub(emoji, " ", text)

    # remove space
    text= " ".join(text.split())

    # change lower word
    text = text.lower()

    # join word
    if not isinstance(text, (float, int)):
        sentence_tokenized = ViTokenizer.tokenize(text)
        text = sentence_tokenized.split()
    else:
        return None
    
    # repair word
    for i in range(len(text)):
        if text[i] in repair_words_list:
            text[i] = repair_words_list[text[i]]

    # add vocabulary
    adding = []
    for key in feature_vocabulary:
        adding += [feature_vocabulary[key]] * text.count(key)
    text = text + adding

    # change into vector word
    word_vectors = []
    for word in text:
        try:
            word_vectors.append(model.wv.get_vector(word))
        except KeyError:
            continue
    try:
        avg_vector = sum(word_vectors) / len(word_vectors)
    except:
        avg_vector = np.zeros((100,))

    # predict text
    sentiment = models.predict(np.array([avg_vector]))[0]
    if sentiment == [0]:
        return "Positive", *list(models.predict_proba([avg_vector])[0])
    if sentiment == [2]:
        return "Negative", *list(models.predict_proba([avg_vector])[0])
    return "Neutral", *list(models.predict_proba([avg_vector])[0])


def PredictList(filename, filetype):
    print(filename, filetype[1][1:])
    data = None
    try:
        data : pd.DataFrame = read_file(filename, filetype[1][1:])
        column_data = data.columns.values.tolist()
        column_data_lower = [x.lower() for x in column_data]

        if not ("opinion" in column_data_lower):
            return None
        
        column_opinion = column_data[column_data_lower.index("opinion")]

        if not ("time" in column_data_lower):
            pos, neg, neu = 0, 0, 0
            for opinion in data[column_opinion]:
                prediction = Predict(opinion)
                if prediction[0] == "Positive":
                    pos += 1
                elif prediction[0] == "Negative":
                    neg += 1
                else:
                    neu += 1
            return False, pos, neg, neu

        column_time = column_data[column_data_lower.index("time")]
        time_list : dict[str, dict[str, int]] = dict()
        total_pos, total_neg, total_neu = 0, 0, 0
        for i in range(len(data)):
            time = data[column_time].loc[i]
            time = datetime.date.strftime(datetime.datetime.strptime(time , FORMAT_DATE).date(), "%d/%m/%Y")
            opinion = data[column_opinion].loc[i]

            if time_list.get(time, None) == None:
                time_list[time] = {"Positive" : 0, "Negative" : 0, "Neutral" : 0}

            pos, neg, neu = 0, 0, 0
            prediction = Predict(opinion)

            if prediction[0] == "Positive":
                pos += 1
            elif prediction[0] == "Negative":
                neg += 1
            else:
                neu += 1
            total_pos += pos
            total_neg += neg
            total_neu += neu
            time_list[time]["Positive"] += pos
            time_list[time]["Negative"] += neg
            time_list[time]["Neutral"] += neu
        time_keys = list(time_list.keys())
        time_keys.sort(key= lambda date: datetime.datetime.strptime(date, "%d/%m/%Y"))
        sorted_time_list = {i : time_list[i] for i in time_keys}

        pos_list = []
        neg_list = []
        neu_list = []

        for time in sorted_time_list:
            pos_list.append(sorted_time_list[time]["Positive"])
            neg_list.append(sorted_time_list[time]["Negative"])
            neu_list.append(sorted_time_list[time]["Neutral"])
            
        return True, total_pos, total_neg, total_neu, time_keys, pos_list, neg_list, neu_list
            
    except Exception as ex:
        print(ex)
        return None
    
if __name__ == "__main__":
    print(Predict("ádasdasd"))