import numpy as np
import pandas as pd
from .read_data import *
from .score import report_score
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

data_dir = './data/'

def predict(model1, model2, test_X):
    train_df, test_df, train_h, train_b, test_h, test_b = read_data(data_dir)

    Lables_1 = ['related', 'unrelated']
    Labels_2 = ['agree', 'disagree', 'discuss', 'unrelated']

    model_1_pred = model1.predict_classes(test_X)
    model_2_pred = model2.predict_classes(test_X)

    pred_1 = [Lables_1[int(pred)] for pred in model_1_pred]
    pred_2 = [Labels_2[int(pred)] for pred in model_2_pred]

    predictions = []

    for idx, row in test_df.iterrows():
        if row.Stance == 'unrelated':
            if pred_1[idx]=='unrelated':
                predictions.append(pred_1[idx])
            else:
                predictions.append(pred_2[idx])
        else:
            predictions.append(pred_2[idx])


    test_df['Predictions'] = predictions
    test_df.to_csv('./output/final_answer.csv')

    report_score(test_df['Stance'].tolist(), predictions)