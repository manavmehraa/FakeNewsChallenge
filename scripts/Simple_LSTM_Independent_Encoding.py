#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd
import csv
import numpy as np
import nltk
#nltk.data.path.append('../data/nltk_data')
from nltk.tokenize import word_tokenize
import keras
from keras.models import Model
from keras.layers import *
from keras.backend import concatenate
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from score import report_score


max_headline_length = 40
max_article_length = 1000
emb_dimension = 100

FIELDNAMES = ['Headline', 'Body ID', 'Stance']
Labels = ['agree', 'disagree', 'discuss', 'unrelated']
di = { Labels[0]:0, Labels[1]:1, Labels[2]:2, Labels[3]:3 }

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

print("Tokenizing Data")

def tokenize_list(sentence_list, max_num_of_words):
    final_list = []
    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        words = [word.lower() for word in tokens if word.isalpha()]
        words = [word for word in words if word not in stopWords]
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

my_tokenizer = Tokenizer(oov_token = 'UNK')
my_tokenizer.fit_on_texts(train_headlines_list_tokenized + test_headlines_list_tokenized + train_article_list_tokenized + test_article_list_tokenized)

train_headlines_seq = my_tokenizer.texts_to_sequences(train_headlines_list_tokenized)
test_headlines_seq = my_tokenizer.texts_to_sequences(test_headlines_list_tokenized)

train_article_seq = my_tokenizer.texts_to_sequences(train_article_list_tokenized)
test_article_seq = my_tokenizer.texts_to_sequences(test_article_list_tokenized)

train_headlines_pad = pad_sequences(train_headlines_seq,padding = 'post', maxlen = max_headline_length)
test_headlines_pad = pad_sequences(test_headlines_seq,padding = 'post', maxlen = max_headline_length)

train_article_pad = pad_sequences(train_article_seq,padding = 'post', maxlen = max_article_length)
test_article_pad = pad_sequences(test_article_seq,padding = 'post', maxlen = max_article_length)

max_Vocabulary_Size = len(my_tokenizer.word_index) + 1
embedding_matrix = np.zeros((max_Vocabulary_Size, emb_dimension))
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

del file

garbage_words = []
for word, i in my_tokenizer.word_index.items():
    embedding_vector = glove_dict.get(word)
    if embedding_vector is not None:
        index = my_tokenizer.word_index[word]
        embedding_matrix[index] = embedding_vector
    else:
        garbage_words.append(word)

train_encoder = OneHotEncoder(sparse=False)
train_labels = train_encoder.fit_transform(train_labels)
test_labels = train_encoder.fit_transform(test_labels)

def get_model(max_Vocab_Size, e_dimension, e_matrix):
    embedding_layer_input1 = Embedding(input_dim = max_Vocab_Size,
                                output_dim = e_dimension,
                                embeddings_initializer = keras.initializers.Constant(e_matrix),
                                trainable=False,
                                )
    input1 = Input((None,), name='lstm1_input')
    x1 = embedding_layer_input1(input1)
    x1 = LSTM(100)(x1)

    embedding_layer_input2 = Embedding(input_dim = max_Vocab_Size,
                                output_dim = e_dimension,
                                embeddings_initializer = keras.initializers.Constant(e_matrix),
                                trainable=False,
                                )
    input2 = Input((None,), name='lstm2_input')
    x2 = embedding_layer_input2(input2)
    x2 = LSTM(100)(x2)
    x = Concatenate()([x1, x2])

    output = Dense(4, activation='softmax')(x)

    model = Model([input1,input2], output)
    print(model.summary())
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['accuracy'])
    return model

model = get_model(max_Vocabulary_Size, emb_dimension, embedding_matrix)
model.fit([train_headlines_pad, train_article_pad], train_labels,
              batch_size=1024,
              epochs=40,
              validation_split=0.2)
loss_accuracy = model.evaluate([test_headlines_pad, test_article_pad], test_labels)
print("Accuracy = " + str(loss_accuracy[1]))

y_predicted = model.predict([test_headlines_pad, test_article_pad])
y_predicted = np.argmax(y_predicted,axis=1)
predicted = [Labels[int(a)] for a in y_predicted]

y_actual = test_stances_df['Stance'].tolist()
actual = [Labels[int(a)] for a in y_actual]
report_score(actual,predicted)


# In[ ]:
