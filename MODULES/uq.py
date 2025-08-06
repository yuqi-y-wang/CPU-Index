import numpy as np
import pandas as pd

from lifelines.utils import concordance_index
from scipy.special import softmax
from scipy.stats import chisquare
from MODULES import metrics_utils

def sorted_patients(df, poi_index, index_train, X_train, X_poi, model, 
                    similarity='nomogram_num_diff', specified_cols=None, 
                    df_og=None, group_num=10):
    '''
    Parameters:
    poi_index: integer, the poi index in reserved
    index_train: list of integers, the indexs for patients in trainset
    index_reserved: list of integers, the indexs for patients in reserved set
    X_reserved: array of the features
    model: instance, the trained model
    similarity: string, choose the similarity method
    Return:
    dataFrame: sorted according to points, and group id assigned
    '''
    if 'nomogram' in similarity:
        poi = df['points'].loc[poi_index]
        poi_predicted = model.predict_survival(X_poi)
        patient_difference = np.abs(poi - df.loc[index_train, 'points'])
    if 'num_diff' in similarity:
        if specified_cols:
            patient_difference += np.sum(np.logical_xor(
                df_og[specified_cols].loc[poi_index].values, 
                df_og[specified_cols].loc[index_train].values),axis=1)
    
    df_patient_difference = patient_difference.to_frame()
    df_patient_difference.columns = ['difference']
    df_patient_difference.sort_values(by='difference', ascending=True, inplace=True)
    df_patient_difference['sim_group'] = np.arange(
        len(df_patient_difference))//(len(df_patient_difference)/group_num)
    return df_patient_difference


def calculate_group_errs(
    df, features, X_poi, df_patient_difference, model, 
    group_num=10, alg_calibration='weighted_sum', alg_err='lse'):
    '''
    Parameters:
    group_num: integer
    df_patient_difference: dataFrame with patient index, points, difference and group assignment
    alg_calibration: string, calibration algorithm
    alg_err: string, error algorithm ['lse', 'chisquare']
    Returns:
    list: error list
    '''
    errs = []
    poi_predicted = model.predict_survival(X_poi)
    
    for g_id in range(group_num):
        g_indexs = df_patient_difference.index[df_patient_difference['sim_group'] == g_id]
        g_predicted = model.predict_survival(df.loc[g_indexs, features])
        if alg_calibration == 'weighted_sum':
            weights = softmax(-df_patient_difference['difference'].loc[g_indexs])
            g_predicted_calibrated = np.average(g_predicted, weights=weights, axis=0)
        if alg_err == 'lse':
            g_err = np.linalg.norm(g_predicted_calibrated-poi_predicted) # lse
        elif alg_err == 'chisquare':
            g_err = chisquare(poi_predicted.flatten(), f_exp=g_predicted_calibrated)[0] # chisquare test
        errs.append(g_err) 
    return errs


def calculate_c_index(errs, group_num=10):
    group_order = list(np.arange(group_num))
    curve_order = np.array(errs).argsort()
    c_poi = concordance_index(group_order, curve_order)
    return c_poi


def decide_warning(df_uq, warning_pred = 0.4, warning_uq = 0.4, 
                   prediction = 'hazard_pred_at_time'):
    df_uq = df_uq.drop(columns=['warning'], errors='ignore')
    df_uq.loc[:, 'warning'] = 1
    inds = list(df_uq.index[
        np.where((df_uq['uq_score']<warning_uq)
                 &(df_uq['hazard_pred_at_time']<warning_pred))[0]])
    inds += list(df_uq.index[
        np.where((df_uq['uq_score']>=warning_uq)
                 &(df_uq['hazard_pred_at_time']<warning_pred+0.1))[0]])
    df_uq.loc[inds, 'warning'] = 0
    return df_uq


def cal_acceptable_rate(df_uq, warning_pred = 0.4, warning_uq = 0.4, 
                        label = 'event', prediction = 'hazard_pred_at_time'):
    df_uq = decide_warning(df_uq, warning_pred, warning_uq, prediction)
    acceptable_rate = metrics_utils.prediction_performance_binary(
        df_uq[label], df_uq['warning'])
    # acceptable_rate = np.sum(df_uq_uncensored['warning']==1)/len(df_uq_uncensored)
    return acceptable_rate

# concordance_index(event_times, predicted_scores, event_observed=None)
