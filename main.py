import numpy as np
import argparse
from utils.build_model1 import *
from utils.build_model2 import *
from utils.predict import predict



parser = argparse.ArgumentParser()
parser.add_argument('--train_feat', type=str,
                    help='Train Features? (y/n')
parser.add_argument('--train_model', type=str,
                    help='Train Model? (y/n')
args = parser.parse_args()


train_y_step_1 = np.load('./trained/train_y_step_1.npy')
test_y_step_1 = np.load('./trained/test_y_step_1.npy')
train_y_step_2 = np.load('./trained/train_y_step_2.npy')
test_y_step_2 = np.load('./trained/test_y_step_2.npy')


if args.train_feat == 'y':
    from utils.features import *
else:
    train_X = np.load('./trained/train_X.npy')
    test_X = np.load('./trained/test_X.npy')

if args.train_model == 'y':
    model1 = build_model1(train_X, test_X, train_y_step_1, test_y_step_1)
    model2 = build_model2(train_X, test_X, train_y_step_2, test_y_step_2)
    predict(model1, model2, test_X)
else:
    from utils.prediction import *