import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sampling_index(df, val=False, sampling_rate=0.2):
    index_train, index_test = train_test_split(df.index.to_list(), 
                                               test_size = sampling_rate, stratify=df['event'])
    index_val = []
    if val:
        index_val, index_test = train_test_split(index_test, 
                                                 test_size = 0.5, 
                                                 stratify=df['event'].loc[index_test])
    return index_train, index_val, index_test


def get_sets_from_index(df, features, index_train, index_val, index_test):
    data_train = df.loc[index_train].reset_index(drop = True)
    data_val = df.loc[index_val].reset_index(drop = True)
    data_test  = df.loc[index_test].reset_index(drop = True)

    # Creating the X, T and E input
    X_train, X_val, X_test = data_train[features].values, data_val[features].values, data_test[features].values
    T_train, T_val, T_test = data_train['time'].values, data_val['time'].values, data_test['time'].values
    E_train, E_val, E_test = data_train['event'].values, data_val['event'].values, data_test['event'].values

    return X_train, X_val, X_test, T_train, T_val, T_test, E_train, E_val, E_test