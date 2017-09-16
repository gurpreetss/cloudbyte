 #!/usr/bin/env python -W ignore
from __future__ import division, print_function, unicode_literals
import warnings
warnings.filterwarnings("ignore")

# Common imports
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from pandas import Series

import os, sys
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

gDataPath = '../data/D11-02/'

def save_img(fig_id, tight_layout=True):
    path = os.path.join("../images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
        plt.savefig(path, format='png', dpi=300)

def read_data():
    data_files = ['D11', 'D12', 'D01', 'D02']
    data = (pd.read_csv(os.path.join(gDataPath,f), sep=';', header=0) for f in data_files)
    concat_data = pd.concat(data, ignore_index=True)
    # Change the column name from chineses to english
    concat_data.columns = ['Transaction Date', 'Customer ID', 'Age', 'Residence Area', 'Product Subclass', 'Product ID', 'Amount', 'Asset', 'Sale Price'] 
    # Handle the time elements for Transaction Date
    concat_data['Transaction Date'] = pd.to_datetime(concat_data['Transaction Date'])
    concat_data.index = concat_data['Transaction Date']
    del concat_data['Transaction Date']
    return concat_data

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def prepare_data(pos_data, product_id):
    item_data = pos_data.filter(items=["Product ID"])[pos_data["Product ID"] == product_id]
    item_freq_data = item_data.groupby(item_data.index.date).count()
    item_freq_data.index = pd.to_datetime(item_freq_data.index)
    item_week_freq_data = item_freq_data.resample('W', how="sum")
    item_week_freq_data["Product ID"] = item_week_freq_data["Product ID"].astype(float)

    # difference data
    weeks = 1
    stationary = difference(item_week_freq_data.values, weeks)
    stationary.index = item_week_freq_data.index[weeks:]
    # check if stationary
    result = adfuller(stationary)
    for key, value in result[4].items():
        stationary = stationary.astype(float)
    return stationary

def train_and_predict_model(data, arima_order, plot=0):
    X = data.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    train_values = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(train_values, order=arima_order)
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        train_values.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    if plot:
        plt.plot(test)
        plt.plot(predictions, color='red')
        save_img("prediction_plot")
        plt.show()
    return error

#Tune the ARIMA hyperparameters for a rolling one-step forecast
# GridSearch and evaluate an ARIMA model for a given order (p,d,q)

def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = train_and_predict_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                        print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA hyperparameters: %s MSE=%.3f' % (best_cfg, best_score))


if __name__=='__main__':
    pos_data = read_data()
    prep_data = prepare_data(pos_data, 4711271000014)
    # evaluate parameters
    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = range(0, 3)
    q_values = range(0, 3)
    evaluate_models(prep_data, p_values, d_values, q_values)

