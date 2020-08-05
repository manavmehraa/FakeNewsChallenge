import tensorflow as tf
from keras.layers import *
from tensorflow import keras
from keras.optimizers import Adam
from tensorflow.keras import layers
from keras.models import Sequential
from keras.initializers import Constant
from tensorflow.keras import initializers
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

def build_model1(train_X, train_y, test_X, test_y):
    initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal', seed=12345)
    
    model = keras.Sequential()
    model.add(layers.Dense(250, input_dim=train_X.shape[1], activation="relu", kernel_initializer=initializer, bias_initializer=initializers.Constant(0.1)))
    model.add(layers.Dropout(0.6))
    model.add(layers.BatchNormalization()) 
    model.add(layers.Dense(50, activation='relu', kernel_initializer=initializer, bias_initializer=initializers.Constant(0.1)))
    model.add(layers.Dense(50, activation='relu', kernel_initializer=initializer, bias_initializer=initializers.Constant(0.1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation="softmax")) 

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    model.fit(train_X, train_y, epochs=90, batch_size=288, validation_split=0.2)
  
    loss, accuracy = model.evaluate(test_X, test_y, verbose=1)
    print('Accuracy on test set: %f' % (accuracy*100))
  
    return model                                                                              