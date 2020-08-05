#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
#nltk.data.path.append('../data/nltk_data')
from nltk.tokenize import word_tokenize
from keras.models import Model

import numpy as np
from scipy.sparse import csr_matrix
from score import report_score

upscale_agree = 1
upscale_disagree = 6

FIELDNAMES = ['Headline', 'Body ID', 'Stance']
Labels = ['agree', 'disagree', 'discuss', 'unrelated']

'''
This list of stopwords has taken from UCLMR’s public GitHub repository: github.com/uclmr/fakenewschallenge
Authors :- Benjamin Riedel, Isabelle Augenstein, Georgios Spithourakis, Sebastian Riedel
'''
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

def read_train_stances_related_unrelated(path, upscale_factor_agree, upscale_factor_disagree):
    df = pd.read_csv(path)
    agree_examples = df[df['Stance'] == Labels[0]]
    for i in range(upscale_factor_agree):
        df = df.append(agree_examples,ignore_index=True)

    disagree_examples = df[df['Stance'] == Labels[1]]
    for i in range(upscale_factor_disagree):
        df = df.append(disagree_examples,ignore_index=True)

    di = { Labels[0]:0, Labels[1]:0, Labels[2]:0, Labels[3]:1 }
    df['Stance'].replace(di, inplace=True)
    return df

def read_stances(path):
    df = pd.read_csv(path)
    di = { Labels[0]:0, Labels[1]:0, Labels[2]:0, Labels[3]:1 }
    df['Stance'].replace(di, inplace=True)
    return df

trainBodiesPath = "../data/train_bodies.csv"
trainStancesPath = "../data/train_stances.csv"
testBodiesPath = "../data/competition_test_bodies.csv"
testStancesPath = "../data/competition_test_stances.csv"

print("Reading Training Data")
train_articles_df,train_dic_articleId_index = read_bodies(trainBodiesPath)
train_stances_df = read_train_stances_related_unrelated(trainStancesPath, upscale_agree, upscale_disagree)
train_labels = train_stances_df[['Stance']].values

print("Reading Testing Data")
test_articles_df,test_dic_articleId_index = read_bodies(testBodiesPath)
test_stances_df = read_stances(testStancesPath)
test_labels = test_stances_df[['Stance']].values


print("Building Vectorizer")
train_article_list = []
test_article_list = []

for index, row in train_stances_df.iterrows():
    train_article_list.append(train_articles_df.iloc[train_dic_articleId_index[int(row['Body ID'])]]['articleBody'])

for index, row in test_stances_df.iterrows():
    test_article_list.append(test_articles_df.iloc[test_dic_articleId_index[int(row['Body ID'])]]['articleBody'])

train_headlines_list = train_stances_df['Headline'].to_list()
test_headlines_list = test_stances_df['Headline'].to_list()


tf_vec = CountVectorizer(stop_words = stopWords, ngram_range = (1,1), max_features = 5000)
tf_vec.fit(train_article_list + train_headlines_list)


tfidf_vec = TfidfVectorizer(stop_words = stopWords, ngram_range = (1,1), max_features = 5000, norm = 'l2')
tfidf_vec.fit(train_article_list + train_headlines_list + test_article_list + test_headlines_list)

train_hline_tf = tf_vec.transform(train_headlines_list)
test_hline_tf = tf_vec.transform(test_headlines_list)

train_hline_tfidf = tfidf_vec.transform(train_headlines_list)
test_hline_tfidf = tfidf_vec.transform(test_headlines_list)

train_article_list_tf = tf_vec.transform(train_article_list)
test_article_list_tf = tf_vec.transform(test_article_list)

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
def generate_final_matrix(hline_tfidf, article_list_tfidf, cos_similarity):
    final_dense = np.zeros(shape=(hline_tfidf.shape[0],10001))

    for x in range(hline_tfidf.shape[0]):
        A = csr_matrix.todense(hline_tfidf[x])
        B = csr_matrix.todense(article_list_tfidf[x])
        C = cos_similarity[x]
        row = np.squeeze(np.c_[A, B, C])
        final_dense[x] = row

    return final_dense

final_train_dense = generate_final_matrix(train_hline_tfidf, train_article_list_tfidf, cos_similarity_train)
final_test_dense = generate_final_matrix(test_hline_tfidf, test_article_list_tfidf, cos_similarity_test)

train_encoder = OneHotEncoder(sparse=False)
train_labels = train_encoder.fit_transform(train_labels)

test_encoder = OneHotEncoder(sparse=False)
test_labels = test_encoder.fit_transform(test_labels)

def build_model(input_dimension):
    model = keras.Sequential()
    model.add(layers.Dense(100,input_dim = input_dimension, activation="relu"))
    model.add(layers.Dropout(0.6))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation="softmax"))

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

model = build_model(final_train_dense.shape[1])

model.fit(final_train_dense, train_labels, epochs=90, batch_size=500)
#evaluate the model
loss, accuracy = model.evaluate(final_test_dense, test_labels, verbose=0)
print('Normal Accuracy: %f' % (accuracy*100))

ypredicted_model_one = model.predict(final_test_dense)
ypredicted_model_one = np.argmax(ypredicted_model_one,axis=1)


# In[4]:


def read_train_stances_three_categories(path, upscale_factor_agree, upscale_factor_disagree):
    df = pd.read_csv(path)
    agree_examples = df[df['Stance'] == Labels[0]]
    for i in range(upscale_factor_agree):
        df = df.append(agree_examples,ignore_index=True)

    disagree_examples = df[df['Stance'] == Labels[1]]
    for i in range(upscale_factor_disagree):
        df = df.append(disagree_examples,ignore_index=True)

    di = { Labels[0]:0, Labels[1]:1, Labels[2]:2, Labels[3]:3 }
    df['Stance'].replace(di, inplace=True)
    df = df[df['Stance'].isin([0, 1, 2])]
    return df


train_stances_df = read_train_stances_three_categories(trainStancesPath, upscale_agree, upscale_disagree)
train_labels = train_stances_df[['Stance']].values

index = []
number_of_test_instance = 0
for i in range(len(ypredicted_model_one)):
    if int(ypredicted_model_one[i]) == 0:
        index.append(i)
        ypredicted_model_one[i] = 0
        number_of_test_instance = number_of_test_instance + 1
    else:
        ypredicted_model_one[i] = 3

test_stances_df = test_stances_df.iloc[index,:]

max_headline_length = 15
max_article_length = 700
max_Vocabulary_Size = 40000
emb_dimension = 100

print("Tokenizing Headline and article body for Model 2")

def tokenize_list(sentence_list, max_num_of_words):
    final_list = []
    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        words = [word.lower() for word in tokens if word.isalpha()]
        final_list.append(words[:max_num_of_words])
    return final_list


#Tokenize the headlines
train_headlines_list = train_stances_df['Headline'].to_list()
test_headlines_list = test_stances_df['Headline'].to_list()

train_headlines_list_tokenized = tokenize_list(train_headlines_list, max_headline_length)
test_headlines_list_tokenized = tokenize_list(test_headlines_list, max_headline_length)


#Tokenize the articles
train_article_list = []
test_article_list = []

for index, row in train_stances_df.iterrows():
    train_article_list.append(train_articles_df.iloc[train_dic_articleId_index[int(row['Body ID'])]]['articleBody'])

for index, row in test_stances_df.iterrows():
    test_article_list.append(test_articles_df.iloc[test_dic_articleId_index[int(row['Body ID'])]]['articleBody'])

train_article_list_tokenized = tokenize_list(train_article_list, max_article_length)
test_article_list_tokenized = tokenize_list(test_article_list, max_article_length)

my_tokenizer = Tokenizer(num_words=max_Vocabulary_Size, oov_token = 'UNK')
my_tokenizer.fit_on_texts(train_headlines_list_tokenized + test_headlines_list_tokenized + train_article_list_tokenized + test_article_list_tokenized)

train_headlines_seq = my_tokenizer.texts_to_sequences(train_headlines_list_tokenized)
test_headlines_seq = my_tokenizer.texts_to_sequences(test_headlines_list_tokenized)

train_article_seq = my_tokenizer.texts_to_sequences(train_article_list_tokenized)
test_article_seq = my_tokenizer.texts_to_sequences(test_article_list_tokenized)

train_headlines_pad = pad_sequences(train_headlines_seq,padding = 'post', maxlen = max_headline_length, truncating='post')
test_headlines_pad = pad_sequences(test_headlines_seq,padding = 'post', maxlen = max_headline_length, truncating='post')

train_article_pad = pad_sequences(train_article_seq,padding = 'post', maxlen = max_article_length, truncating='post')
test_article_pad = pad_sequences(test_article_seq,padding = 'post', maxlen = max_article_length, truncating='post')

train_seq = np.concatenate((train_headlines_pad,train_article_pad),axis=1)
test_seq = np.concatenate((test_headlines_pad,test_article_pad),axis=1)


embedding_matrix_length = len(my_tokenizer.word_index) + 1
embedding_matrix = np.zeros((embedding_matrix_length, emb_dimension))
embedding_matrix[0] = np.random.random((1, emb_dimension))
embedding_matrix[1] = np.random.random((1, emb_dimension))

glove_dict = {}

glove_data = '../data/glove.6B.100d.txt'
file = open(glove_data)
for l in file:
    values = l.split()
    word = values[0]
    value = np.asarray(values[1:], dtype='float32')
    glove_dict[word] = value

file.close()

garbage_words = []
for word, i in my_tokenizer.word_index.items():
    embedding_vector = glove_dict.get(word)
    if embedding_vector is not None:
        index = my_tokenizer.word_index[word]
        embedding_matrix[index] = embedding_vector
    else:
        garbage_words.append(word)

train_labels = train_stances_df[['Stance']].values

train_encoder = OneHotEncoder(sparse=False)
train_labels = train_encoder.fit_transform(train_labels)


def build_second_model(e_matrix_length, e_dimension, e_matrix):
    model = keras.Sequential()
    model.add(layers.Input(shape=(715,), name='input_layer'))
    model.add(layers.Embedding(input_dim = e_matrix_length,
                                output_dim = e_dimension,
                                embeddings_initializer=keras.initializers.Constant(e_matrix),
                                trainable=True,
                                ))
    model.add(layers.Dense(64, input_dim = e_dimension, activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(3, activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return model

model = build_second_model(embedding_matrix_length, emb_dimension, embedding_matrix)
model.fit(train_seq, train_labels, epochs=20, batch_size=500, validation_split=0.1)

ypredicted_model_two = model.predict(test_seq)
ypredicted_model_two = np.argmax(ypredicted_model_two,axis=1)

ypredicted_model_two = [int(a) for a in ypredicted_model_two]

index_model_two = 0

for i in range(len(ypredicted_model_one)):
    if ypredicted_model_one[i] == 0:
        ypredicted_model_one[i] = ypredicted_model_two[index_model_two]
        index_model_two = index_model_two + 1
    else:
        continue


predicted = [Labels[a] for a in ypredicted_model_one]
df = pd.read_csv(testStancesPath)
actual = df['Stance'].tolist()
report_score(actual,predicted)


# In[ ]:
