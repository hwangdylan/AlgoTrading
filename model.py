from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import pandas
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
from data_processor import *


class Model():

    def __init__(self, BS, TS):
        """
        initialize the model
        """
        self.model = Sequential()
        self.BS = BS #batch size
        self.TS = TS #time steps

    def make_model(self, input_shape, lr, TIME_STEPS, BATCH_SIZE):
    #     model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, train_data.shape[1]), dropout=0.0, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform', return_sequences=True))
        self.model.add(LSTM(100, input_shape=input_shape, dropout=0.0, recurrent_dropout=0.0, kernel_initializer='random_uniform', return_sequences=True))
        print("HI")
        self.model.add(Dropout(0.4))
        self.model.add(LSTM(60, dropout=0.0))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(20,activation='relu'))
        self.model.add(Dense(4,activation='sigmoid'))
        optimizer = optimizers.RMSprop(lr=lr)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        return self.model

    def train(self, train_data, train_labels):
        OUTPUT_PATH = ""
        csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'log' + '.log'), append=True)
        # history = self.model.fit(x_t, y_t, epochs=2, verbose=2, batch_size=BATCH_SIZE,
        #                     shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),trim_dataset(y_val, BATCH_SIZE)), callbacks=[csv_logger])
        history = self.model.fit(train_data, 
                            train_labels, 
                            epochs=10, 
                            verbose=2, 
                            batch_size=self.BS,
                            shuffle=False, 
                            callbacks=[csv_logger])
        return history
    def make_prediction(self, test_data, TIME_STEPS, BATCH_SIZE):
        """
        the test_data passed in is the matrix of features with shape 
        (T, D) where T is the number of time steps there are and D is 
        the number of features
        """
        scale = MinMaxScaler()
        temp = scale.fit_transform(test_data)
        print("shape1:", temp.shape)
        temp, _ = build_timeseries(temp, TIME_STEPS)
    #     temp = trim_dataset(temp, BATCH_SIZE)
        print("shape:", temp.shape)
        prediction = self.model.predict(temp)
        pred = (prediction* scale.data_range_[0]) + scale.data_min_[0]
        print("prediction shape:", pred.shape)
        # figure(num=None, figsize=(15, 6), dpi=70, facecolor='w', edgecolor='k')
        # plt.plot(df.reset_index()['Date'].values[:len(pred)],pred)
        # plt.plot(df.reset_index()['Date'].values[:len(pred)],df['Close'].values[TIME_STEPS:TIME_STEPS +  len(pred)])

        # plt.tight_layout()
        # plt.grid()
        # plt.show()
        return pred
    def predict_full_sequence(self, data):
        curr_frame = data[:self.TS + 1, :]
        predictions = []
        for i in range(500):
            print("Iteration: ", i)
            predictions.append(self.make_prediction(curr_frame, self.TS, self.BS))
            curr_frame = curr_frame[1:]
            curr_frame = np.concatenate([curr_frame, predictions[-1]], axis=0)
        return predictions
