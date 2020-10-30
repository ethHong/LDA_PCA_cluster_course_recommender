import pandas as pd
import json
import re
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm, tqdm_pandas
import matplotlib.pyplot as plt
import seaborn as sns


print ("Loading data...")
df = pd.read_csv("courses_data.csv")
df = df.loc[df["Year"].isin(["2018", "2019", "2020"])]
print ("Data Loaded!...")

print ("Load JD of Google..")
with open('linkedin_JD_Google.json') as json_file:
    json_data = json.load(json_file)
print ("Load JD of BCG..")
with open('linkedin_JD_Boston Consulting Group.json') as json_file:
    json_data2 = json.load(json_file)
print ("Load JD of JP Morgan..")
with open('linkedin_JD_jpmorgan.json') as json_file:
    json_data3 = json.load(json_file)

print ("Transforming target doc and collections...")

UIC = df.loc[df["Division"].apply(lambda x: x.startswith("언더우드국제대학"))]

collection = list(json_data.values())
target_doc =[keyword for bag in collection for keyword in bag]
position  =[i for i in list(json_data.keys())]

collection2 = list(json_data2.values())
target_doc2 =[keyword for bag in collection2 for keyword in bag]
position2  =[i for i in list(json_data2.keys())]

collection3 = list(json_data3.values())
target_doc3 =[keyword for bag in collection3 for keyword in bag]
position3  =[i for i in list(json_data3.keys())]


print ("Filtering Stopwords from target docs...")
from stopword_filter import stem_word, filter_stopwords, cleanse, sw, stemmer, cleanse_df, tfidf, filter_more

def process_filter(target):
    target = " ".join([i.lower() for i in target])
    target = filter_stopwords(target, sw)
    target = cleanse(target)
    target = target.split()

    return target

target_doc = process_filter(target_doc)
target_doc2 = process_filter(target_doc2)
target_doc3 = process_filter(target_doc3)

print ("Stopword Filter from target docs complete!")

print ("Cleansing Dataframe")
data = cleanse_df(UIC)
data.drop_duplicates(subset='Course_Name', keep="last")


additional = filter_more(data, threshold = 0.1, par = "avg")["words"].values

print ("Additionally filtering {} Words based on TF-IDF...".format(len(additional)))

def refilter(lst):
    return [i for i in lst if i not in additional]

data["tokenized"] = data["tokenized"].apply(lambda x : refilter(x))
data["tokenized"] = data["tokenized"].apply(lambda x: "None" if x==[] else x)
data = data.loc[data["tokenized"]!="None"]

print ("filtered!")

print("""=========================
=========================""")

print ("Processing LDA Training...")

from LDAModel import train_lda, jsd

num_topics = int(input("Put num_topics"))
passes = 25

print ("Training LDA...")
dictionary,corpus,lda = train_lda(data, num_topics = num_topics, passes = 25)

print ("showing topics: ")
lda.show_topics(num_topics=10, num_words=10)

print ("Criating Topic_Dist_Matrix")

doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
topic_dist = pd.DataFrame(doc_topic_dist, columns = ["topic{}".format(i) for i in range(1,num_topics+1 )])
topic_word_dist_matrix = pd.concat([data[["Course_Name", "tokenized", "div"]].reset_index(drop = True), topic_dist], axis = 1)

words = []
for i in data["tokenized"].values:
    for j in i:
        words.append(j)

words = list(set(words))

def wordcount(w, doc):
    i=0
    for v in doc:
        if v==w:
            i+=1
    return i

import tqdm
from tqdm import tqdm_gui

for i in tqdm_gui(range(0, len(words))):
    w = words[i]
    topic_word_dist_matrix[w] = topic_word_dist_matrix["tokenized"].apply(lambda x: wordcount(w, x))

new = topic_word_dist_matrix.iloc[:, :num_topics+3]
new = new.drop_duplicates(subset='Course_Name', keep="last")

print ("Only include courses with more tha 3 words describtion")
def wordcount(x):
    return len(list(set(x)))

new = new.loc[new["tokenized"].map(wordcount)>=3]

print ("Data Processing Complete!")

#new.to_csv("processed_courses_data_{}topic.csv".format(num_topics), index = False)
print ("Exported!")

