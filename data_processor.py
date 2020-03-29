from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
import yfinance as yf

def setup_train_test_split(ticker, train_prop = 0.8):
    #setup training data 
    df = setup_df(ticker)
    prices = get_data_matrix(df)
    train_data, test_data = train_test_split(prices,train_size=train_prop, test_size= 1- train_prop, shuffle=False)
    num_train_sample = train_data.shape[0]
    num_test_sample = test_data.shape[0]
    #dataframe of the dates for training sample
    train_dates = df.reset_index().iloc[:num_train_sample]["Date"].values
    test_dates = df.reset_index().iloc[num_train_sample:]["Date"].values
    return train_data, test_data, num_train_sample, num_test_sample, train_dates, test_dates

def build_timeseries(mat, TIME_STEPS, y_col_index = None):
    """
    Formats the data into time steps and gets the test labels

    y_col_index is the index of column that would act as output column
    total number of time-series samples would be len(mat) - TIME_STEPS
    if y_col_index is None then it takes the entire row
    D number of features
    """
    dim_0 = mat.shape[0] - TIME_STEPS
    D = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, D))
    if y_col_index:
        y = np.zeros((dim_0, ))
    else:
        y = np.zeros((dim_0, D))
    
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        if y_col_index:
            y[i] = mat[TIME_STEPS + i, y_col_index]
        else:
            y[i] = mat[TIME_STEPS + i, :]

    print("length of time-series i/o",x.shape,y.shape)
    return x, y

def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    This is only needed if you use stateful lstm
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    print("number of rows dropped", no_of_rows_drop)
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat
def setup_df(ticker):
    """
    Gets the ticker and then returns the dataframe
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period="max")
    del df['Dividends']
    del df['Stock Splits']
    return df 
def get_data_matrix(df):
    """
    returns the"Open", "High", 'Low', "Close" columns
    in terms of a numpy array
    """
    return df[["Open", "High", 'Low', "Close"]].to_numpy()
