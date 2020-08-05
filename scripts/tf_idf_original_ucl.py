#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from score import report_score
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.callbacks import EarlyStopping



FIELDNAMES = ['Headline', 'Body ID', 'Stance']
Labels = ['agree', 'disagree', 'discuss', 'unrelated']

stopWords = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]

def read_bodies(path):
    df = pd.read_csv(path)
    Dict = {}
    for index, row in df.iterrows():
        Dict[int(row['Body ID'])] = index
    return df,Dict

def read_stances(path):
    df = pd.read_csv(path)
    di = { Labels[0]:0, Labels[1]:1, Labels[2]:2, Labels[3]:3 }
    df['Stance'].replace(di, inplace=True)
    return df

trainBodiesPath = "../data/train_bodies.csv"
trainStancesPath = "../data/train_stances.csv"
testBodiesPath = "../data/competition_test_bodies.csv"
testStancesPath = "../data/competition_test_stances.csv"

print("Reading Training Data")
train_articles_df,train_dic_articleId_index = read_bodies(trainBodiesPath)
train_stances_df = read_stances(trainStancesPath)
train_labels = train_stances_df[['Stance']].values

print("Reading Testing Data")
test_articles_df,test_dic_articleId_index = read_bodies(testBodiesPath)
test_stances_df = read_stances(testStancesPath)
test_labels = test_stances_df[['Stance']].values

print("Building Vectorizers")
train_article_list = []
test_article_list = []

for index, row in train_stances_df.iterrows():
    train_article_list.append(train_articles_df.iloc[train_dic_articleId_index[int(row['Body ID'])]]['articleBody'])

for index, row in test_stances_df.iterrows():
    test_article_list.append(test_articles_df.iloc[test_dic_articleId_index[int(row['Body ID'])]]['articleBody'])
    
train_headlines_list = train_stances_df['Headline'].to_list()
test_headlines_list = test_stances_df['Headline'].to_list()

train_article_list_temp = train_article_list
train_headlines_list_temp = train_headlines_list

tf_vectorizer = CountVectorizer(stop_words = stopWords, ngram_range = (1,1), max_features = 5000)
tf_vectorizer = tf_vectorizer.fit(train_article_list_temp + train_headlines_list_temp)
tf_vectorizer_data = tf_vectorizer.transform(train_article_list_temp + train_headlines_list_temp)

tf_vec = TfidfTransformer(use_idf=False).fit(tf_vectorizer_data)

tfidf_vec = TfidfVectorizer(stop_words = stopWords, ngram_range = (1,1), max_features = 5000, norm = 'l2')
tfidf_vec.fit(train_article_list + train_headlines_list + test_article_list + test_headlines_list)

train_hline_tf = tf_vec.transform(tf_vectorizer.transform(train_headlines_list))
test_hline_tf = tf_vec.transform(tf_vectorizer.transform(test_headlines_list))

train_hline_tfidf = tfidf_vec.transform(train_headlines_list)
test_hline_tfidf = tfidf_vec.transform(test_headlines_list)


train_article_list_tf = tf_vec.transform(tf_vectorizer.transform(train_article_list))
test_article_list_tf = tf_vec.transform(tf_vectorizer.transform(test_article_list))

train_article_list_tfidf = tfidf_vec.transform(train_article_list)
test_article_list_tfidf = tfidf_vec.transform(test_article_list)

def generate_cos_similarity_matrix(hline_tfidf, article_list_tfidf):
    cos_similarity = np.zeros(shape=(hline_tfidf.shape[0],1))
    for x in range(hline_tfidf.shape[0]):
        A = csr_matrix.todense(hline_tfidf[x])
        B = csr_matrix.todense(article_list_tfidf[x])
        cos_similarity[x] = cosine_similarity(A, B)[0]
    return cos_similarity

cos_similarity_train = generate_cos_similarity_matrix(train_hline_tfidf, train_article_list_tfidf)
    
cos_similarity_test = generate_cos_similarity_matrix(test_hline_tfidf, test_article_list_tfidf)

print("Building final train and test matrix")
def generate_final_matrix(hline_tf, article_list_tf, cos_similarity):
    final_dense = np.zeros(shape=(hline_tf.shape[0],10001))

    for x in range(hline_tf.shape[0]):
        A = csr_matrix.todense(hline_tf[x])
        B = csr_matrix.todense(article_list_tf[x])
        C = cos_similarity[x]
        row = np.squeeze(np.c_[A, B, C])
        final_dense[x] = row
    
    return final_dense

final_train_dense = generate_final_matrix(train_hline_tf, train_article_list_tf, cos_similarity_train)
final_test_dense = generate_final_matrix(test_hline_tf, test_article_list_tf, cos_similarity_test)

train_encoder = OneHotEncoder(sparse=False)
train_labels = train_encoder.fit_transform(train_labels)

test_encoder = OneHotEncoder(sparse=False)
test_labels = test_encoder.fit_transform(test_labels)

def build_model(input_dimension):
    model = keras.Sequential()
    model.add(layers.Dense(100,input_dim = input_dimension, activation="relu"))
    model.add(layers.Dropout(0.6))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(4, activation="softmax"))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
    
model = build_model(final_train_dense.shape[1])

model.fit(final_train_dense, train_labels, epochs=90, batch_size=500, validation_split = 0.1)
#evaluate the model
loss, accuracy = model.evaluate(final_test_dense, test_labels, verbose=0)
print('Normal Accuracy: %f' % (accuracy*100))

y_predicted = model.predict_classes(final_test_dense)
predicted = [Labels[int(a)] for a in y_predicted]
y_actual = test_stances_df['Stance'].tolist()
actual = [Labels[int(a)] for a in y_actual]
report_score(actual,predicted)


target_classes = ['agree', 'disagree', 'discuss', 'unrelated']
print(classification_report(y_actual, y_predicted, target_names=target_classes))


# In[ ]:




