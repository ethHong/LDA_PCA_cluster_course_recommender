import pandas as pd
import json
import re
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm, tqdm_pandas
from ast import literal_eval
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

num_topics = int(input("Put your topic number"))
passes = 25


df = pd.read_csv("processed_courses_data_{}topic.csv".format(num_topics))

df["tokenized"] = df["tokenized"].apply(literal_eval)


print ("Loading Target Documents")

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

from LDAModel import train_lda

dictionary,corpus,lda = train_lda(df, num_topics = num_topics, passes = 25)

print ("Baseline Cluster")

X = df.iloc[:, 3:]
kmeans = KMeans(n_clusters=num_topics)
kmeans.fit(X)

df["cluster"] = kmeans.labels_
clustered = df[["Course_Name", "tokenized", "div", "cluster"]]

def predict_new(target_doc, dictionary, lda, model):
    query_bow = dictionary.doc2bow(target_doc)
    new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=query_bow)]).reshape(1, -1).astype(np.float)
    result = model.predict(new_doc_distribution)[0]
    return result, new_doc_distribution[0]

google = predict_new(target_doc, dictionary, lda, kmeans)[0]
new_doc_distribution1 = predict_new(target_doc, dictionary, lda, kmeans)[1]

BCG = predict_new(target_doc2, dictionary, lda, kmeans)[0]
new_doc_distribution2 = predict_new(target_doc2, dictionary, lda, kmeans)[1]

JP_morgan = predict_new(target_doc3, dictionary, lda, kmeans)[0]
new_doc_distribution3 = predict_new(target_doc3, dictionary, lda, kmeans)[1]

print ("Adding Prediction to the Clustered df...")

clustered.loc[len(clustered)] = ["google", target_doc, "google"] + [google]
clustered.loc[len(clustered)] = ["BCG", target_doc2, "BCG"] + [BCG]
clustered.loc[len(clustered)] = ["JP_morgan", target_doc3, "JP_morgan"]+ [JP_morgan]

print ("Exporting...")
clustered[["Course_Name", "tokenized", "div",  "cluster"]].to_csv("Baseline_Clustering_{}topic.csv".format(num_topics), index = False)
clustered[["Course_Name", "tokenized", "div",  "cluster"]].to_excel("Baseline_Clustering_{}topic.xlsx".format(num_topics), index = False)

print ("PCA Clustering")
print ("PCA Scree Plot")

df = pd.read_csv("processed_courses_data_{}topic.csv".format(num_topics))
df["tokenized"] = df["tokenized"].apply(literal_eval)

df.loc[len(df)] = ["google", target_doc, "google"]+ list(new_doc_distribution1)
df.loc[len(df)] = ["BCG", target_doc2, "BCG"]+ list(new_doc_distribution2)
df.loc[len(df)] = ["JP_morgan", target_doc3, "JP_morgan"]+ list(new_doc_distribution3)

X = df.iloc[:, 3:]
n_components = len(X.columns)
pca = PCA(n_components=n_components)
pca.fit(X)

def scree(pca, n):
    sns.pointplot(x = np.arange(0, n), y = pca.explained_variance_)
    plt.xlabel("PCs")
    plt.ylabel("Variance of PC")
    plt.show()

scree(pca, num_topics)

n_components = int(input("Put # PCs of your choice"))

pca = PCA(n_components=n_components)
pca_X = pca.fit_transform(X)

kmeans = KMeans(n_clusters=num_topics)
kmeans.fit(pca_X)

df["cluster"] = kmeans.labels_
pcaclustered = df[["Course_Name", "tokenized", "div",  "cluster"]]
pcaclustered.to_csv("PCA_Clustered_{}topic.csv".format(num_topics), index = False)
pcaclustered.to_excel("PCA_Clustered_{}topic.xlsx".format(num_topics), index = False)
print ("File Exported!")

google = int(pcaclustered.loc[pcaclustered["Course_Name"]=="google"].cluster)
BCG = int(pcaclustered.loc[pcaclustered["Course_Name"]=="BCG"].cluster)
JP_morgan = int(pcaclustered.loc[pcaclustered["Course_Name"]=="JP_morgan"].cluster)

print ("Google Group")
print(pcaclustered.loc[pcaclustered["cluster"]==google])
print ("BCG Group")
print(pcaclustered.loc[pcaclustered["cluster"]==BCG])
print ("JP Morgan Group")
print(pcaclustered.loc[pcaclustered["cluster"]==JP_morgan])

