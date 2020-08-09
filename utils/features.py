import nltk
import regex as re
import numpy as np
import pandas as pd
from tqdm import tqdm
from .read_data import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

_wnl = nltk.WordNetLemmatizer()

'''
if not downloaded already
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
'''

### baseline helper functions ###

def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features

def tokenize(text):
  pattern = re.compile("[^a-zA-Z0-9 ]+")
  stop_words = set(stopwords.words('english'))
  text = pattern.sub('', text.replace('\n', ' ').replace('-', ' ').lower())
  text = [word for word in word_tokenize(text) if word not in stop_words]
  text = [_wnl.lemmatize(t) for t in text]
  return text


def getDocTermFrequency(document, ngram = 1):
    tokens = tokenize(document)
    tf_dict = defaultdict(float)
    for i in range(len(tokens)):
        key_array = []
        for j in range(ngram):
            if i+j < len(tokens):
                key_array.append(tokens[i+j])
                if(len(key_array) == 1):
                    tf_dict[key_array[0]] += 1.0
                else:
                    tf_dict[tuple(key_array)] += 1.0
    return tf_dict 


def getIdfDict(corpus):
  total_docs = len(corpus)
  df_dict = defaultdict(float)
  idf_dict = defaultdict(float)
  
  for document in corpus:
    tokens = tokenize(document)
    for token in set(tokens):
      df_dict[token] += 1.0
  
  for word in df_dict:
    idf_dict[word] = np.log((1.0 + total_docs) / (1.0 + df_dict[word])) + 1.0
      
  return idf_dict


def loadGloveModel():
    f = open(data_dir + "glove.6B.50d.txt", "rb")
    model = {}
    for line in f:
        splitLine = line.split()
        word = str(splitLine[0]).split("'")[1]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

### baseline features ###

def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake', 'fraud', 'hoax', 'false', 'deny', 'denies', 'not', 'despite', 'nope', 'doubt', \
        'doubts', 'bogus', 'debunk', 'pranks', 'retract'
    ]

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake', 'fraud', 'hoax', 'false', 'deny', 'denies', 'not', 'despite', 'nope', 'doubt', \
        'doubts', 'bogus', 'debunk', 'pranks', 'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))
    return X

### hand-crafted features ###



def getGloveVector(document):
    tfidf_dict = defaultdict(float)
    tf_dict = getDocTermFrequency(document)

    for word in tf_dict:
      tfidf_dict[word] = tf_dict[word] * idf_dict[word]

    glove_vector = np.zeros(glove_model['glove'].shape[0])
    if np.sum(list(tfidf_dict.values())) == 0.0:  # edge case: document is empty
        return glove_vector

    for word in tfidf_dict:
        if word in glove_model:
            glove_vector += glove_model[word] * tfidf_dict[word]
    glove_vector /= np.sum(list(tfidf_dict.values()))
    return glove_vector

def divergence(h, b):
  sigma = 0.0
  for i in range(h.shape[0]):
      sigma += h[i] * np.log(h[i] / b[i])
  return sigma

def kl_divergence(headline, body, eps=0.1):
#reference: https://github.com/bmaulana/fake-news-challenge/blob/master/main.py
    X = []  

    for h, b in zip(headline, body):
      tf_headline = getDocTermFrequency(h)
      tf_body = getDocTermFrequency(b)
    
      words = set(tf_headline.keys()).union(set(tf_body.keys()))
      vec_headline, vec_body = np.zeros(len(words)), np.zeros(len(words))
      i = 0
      
      for word in words:
          vec_headline[i] += tf_headline[word]
          vec_body[i] = tf_body[word]
          i += 1
    
      lm_headline = vec_headline + eps
      lm_headline /= np.sum(lm_headline)
      lm_body = vec_body + eps
      lm_body /= np.sum(lm_body)
      X.append(divergence(lm_headline, lm_body))
    
    return X


def magnitude(vec):
    return np.sqrt(np.sum(np.square(vec)))

def cosine_similarity_changed(headline, body):
#reference: https://github.com/bmaulana/fake-news-challenge/blob/master/main.py
  headline_vector = getGloveVector(headline)
  body_vector = getGloveVector(body)

  if magnitude(headline_vector) == 0.0 or magnitude(body_vector) == 0.0:
    return 0.0

  dot_product = 0.0
  for i in range(headline_vector.shape[0]):
    dot_product += headline_vector[i] * body_vector[i]

  similarity = dot_product / (magnitude(headline_vector) * magnitude(body_vector))

  return similarity

def tfidf_cosine(train_df, test_df):
  
  feature_dimension = 10047
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

      
  train_headlines_list = train_df['Headline'].to_list()
  test_headlines_list = test_df['Headline'].to_list()

  train_article_list = train_df['articleBody'].to_list()
  test_article_list = test_df['articleBody'].to_list()

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


  cos_similarity_train = np.zeros(shape=(train_hline_tfidf.shape[0],1))

  for x in range(train_hline_tfidf.shape[0]):
      A = csr_matrix.todense(train_hline_tfidf[x])
      B = csr_matrix.todense(train_article_list_tfidf[x])
      cos_similarity_train[x] = cosine_similarity(A,B)

  cos_similarity_test = np.zeros(shape=(test_hline_tfidf.shape[0],1))

  for x in range(test_hline_tfidf.shape[0]):
      A = csr_matrix.todense(test_hline_tfidf[x])
      B = csr_matrix.todense(test_article_list_tfidf[x])
      cos_similarity_test[x] = cosine_similarity(A,B)
  
  final_train_dense = np.zeros(shape=(train_hline_tfidf.shape[0],feature_dimension))
  final_test_dense = np.zeros(shape=(test_hline_tfidf.shape[0],feature_dimension))

  for x in range(train_hline_tfidf.shape[0]):
      A = csr_matrix.todense(train_hline_tfidf[x])
      B = csr_matrix.todense(train_article_list_tfidf[x])
      C = cos_similarity_train[x]
      D = np.array(wd_overlap[x])
      E = np.array(ref_features[x]).reshape(1,15)
      F = np.array(pol_features[x]).reshape(1,2)
      G = np.array(hd_features[x]).reshape(1, 26)
      H = np.array(kl_features[x])
      I = np.array(cos_features[x])
      row = np.squeeze(np.c_[A, B, C, D, E, F, G, H, I])
      final_train_dense[x] = row


  for x in range(test_hline_tfidf.shape[0]):
      A = csr_matrix.todense(test_hline_tfidf[x])
      B = csr_matrix.todense(test_article_list_tfidf[x])
      C = cos_similarity_test[x]
      D = np.array(wd_overlap_test_features[x])
      E = np.array(ref_test_features[x]).reshape(1,15)
      F = np.array(pol_test_features[x]).reshape(1,2)
      G = np.array(hd_test_features[x]).reshape(1, 26)
      H = np.array(kl_test_features[x])
      I = np.array(cos_test_features[x])
      row = np.squeeze(np.c_[A, B, C, D, E, F, G, H, I])
      final_test_dense[x] = row
  
  return final_train_dense, final_test_dense 



data_dir = './data/'
#reading data and upscaling
train_df, test_df, train_h, train_b, test_h, test_b = read_data(data_dir)

#upscaling
agree_df = train_df[train_df['Stance']=='agree']
disagree_df = train_df[train_df['Stance']=='disagree']

train_df = pd.concat([train_df, agree_df, disagree_df, disagree_df, disagree_df, disagree_df, disagree_df, disagree_df])
train_h, train_b = train_df['Headline'].to_list(), train_df['articleBody'].to_list()
test_h, test_b = test_df['Headline'].to_list(), test_df['articleBody'].to_list()
glove_model = loadGloveModel()


#build training features
wd_overlap = word_overlap_features(train_h, train_b)
ref_features = refuting_features(train_h, train_b)
pol_features = polarity_features(train_h, train_b)
hd_features = hand_features(train_h, train_b)
idf_dict = getIdfDict(train_h+train_b)
kl_features = kl_divergence(train_h, train_b)

cos_features = []
for h, b in tqdm(zip(train_h, train_b)):
    cos_features.append(cosine_similarity_changed(h,b))


#build test features
wd_overlap_test_features = word_overlap_features(test_h, test_b)
ref_test_features = refuting_features(test_h, test_b)
pol_test_features = polarity_features(test_h, test_b)
hd_test_features = hand_features(test_h, test_b)
kl_test_features = kl_divergence(test_h, test_b)

cos_test_features=[]
for h, b in tqdm(zip(test_h, test_b)):
    cos_test_features.append(cosine_similarity_changed(h,b))

train_X, test_X = tfidf_cosine(train_df, test_df)
np.save('./trained/train_X', train_X)
np.save('./trained/test_X', test_X)