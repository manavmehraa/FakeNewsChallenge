## Fake News Challenge - MSCI641 Spring'20
## Rajbir's Branch
This repository contains the code for some of the experiments done for Fake News Challenge-1(FNC-1). This project is done as part of MSCI-641 course project in Spring2020.

Credit:
* Rajbir Singh (rsrajbir@uwterloo.ca)
* Manav Mehra (m3mehra@uwaterloo.ca)

## Getting Started
This code has been run on the Compute Canada servers with 1 Tesla T4 GPU, 8 CPUcores, and a memory size of 30GB. The FNC dataset is included in the ``data`` folder. Code for models is present in ``scripts`` folder. Some scripts require pre-trained 100-d word embedding vectors from Stanford Glove dataset which is not uploaded due to its large size. Before running any model, please download ``glove.6B.zip`` file from [Glove Dataset](https://nlp.stanford.edu/projects/glove/). Copy the ``glove.6B.100d.txt`` file from zipped folder to ``data`` folder of our repository. Then you are all set to go.

## Main Dependencies
    Keras==2.4.3        
    nltk==3.5
    numpy==1.19.0
    pandas==1.0.3
    scikit-learn==0.23.0
    scipy==1.4.1
    tensorflow==2.2.0

## Installing Dependencies
    pip install -r requirements.txt
## File Structure
.
 * [data](./data)
 * [scripts](./scripts)
   * [tf_idf_original_ucl.py](./scripts/tf_idf_original_ucl.py)
   * [tf_idf_experimental.py](./scripts/tf_idf_experimental.py)
   * [tfidf_Word_Vectors_2Step.py](./scripts/tfidf_Word_Vectors_2Step.py)
   * [Bidirectional_LSTM.py](./scripts/Bidirectional_LSTM.py)
   * [Simple_LSTM_Independent_Encoding.py](./scripts/Simple_LSTM_Independent_Encoding.py)
   * [score.py](./scripts/score.py)

There are total 6 files in ``scripts`` folder. 5 of them are the python scripts to run the corresponding models as described below. Last file is ``score.py`` when helps to calculate FNC score of the predictions.
* ``tf_idf_original_ucl.py`` : Our own implementation of [UCL paper](https://arxiv.org/abs/1707.03264) with Tf vectors of headline and article body.
* ``tf_idf_experimental.py`` : Implementation of tf-idf approach with tf-idf vectors of headline and article body.
* ``tfidf_Word_Vectors_2Step.py`` : Implementation of 2 step classifier approach using tfidf and word vectors
* ``Bidirectional_LSTM.py`` : Implementation of 1 step Bidirectional LSTM.
* ``Simple_LSTM_Independent_Encoding.py`` : Implementation of 1 step LSTM with independent encoding of headline and body.


## Scoring Your Classifier
First of all, create python3 virtual environment and download our github repository. Then install all the requirements. Download the 100-d word embeddings vectors file from Stanford Glove dataset as mentioned above and put it in ``data`` folder. Further steps to run tf-idf implementation are given below. The same procedure can be used to run any other model.

    cd scripts
    python tf_idf_original_ucl.py
