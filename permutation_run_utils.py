import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, auc
import time

from importlib import reload

import mysurvival
from mysurvival.models.mlp import MLP_reg_Model
from mysurvival.utils.metrics import concordance_index as mysurv_c_index
from mysurvival.utils.display import integrated_brier_score

import MODULES
from MODULES import preprocess_dict
from MODULES import metrics_utils
from MODULES.surv_sampling import sampling_index, get_sets_from_index
from MODULES.uq import sorted_patients, calculate_group_errs, calculate_c_index, \
    cal_acceptable_rate, decide_warning
from MODULES.surv_utils import cal_time_bin_index
from MODULES.pe_functions import position_encoding

from decimal import Decimal

def make_df_stats(dir):
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)
    if os.path.exists(dir+'stats.csv'):
        df_stats = pd.read_csv(dir+'stats.csv', index_col=0)
    else:
        df_stats = pd.DataFrame(columns=['c-index', 'ibs', 
                                        'AUC', 'AUC_based-ACC', 
                                        'AUC_based-FPR', 'AUC_based-FNR',
                                        'Youden_based-ACC', 'Youden_based-FPR', 
                                        'Youden_based-FNR'])
    return df_stats

def make_df(result_dir, settings, combination):
    if os.path.exists(result_dir):
            df_results = pd.read_csv(result_dir, index_col=0)
    else:
        df_results = pd.DataFrame(columns = ['img_name', 'split', 'uq_score', 'event', 
                                            'time', 'Lung_Rads_merge', 'label'] + \
                                list(range(settings['time_bins']-1)))
        
    ###########################################make dataset################################
    df_ehr = pd.read_csv(preprocess_dict.df_dir,index_col=0)
    df_radimoics = pd.read_csv(preprocess_dict.df_rad_dir_skim, index_col=0)

    EHR_cols = ['CPT_CODE', 'AGE_AT_LCS_EXAM', 'SEX', 'RACE', 'TOBACCO_USED_YEARS',
                    'TOBACCO_PACKS_PER_DAY', 'YEARS_SINCE_QUITTING']

    nodule_cols = ['max_lesion_size', 'mean_lesion_size', 
                'lesion_locations_0', 'lesion_locations_1', 'lesion_locations_2', 
                'lesion_locations_3', 'lesion_locations_4', 'lesion_locations_5', 
                'lesion_locations_6', 'lesion_locations_7', 'lesion_locations_8', ]

    feature_cols = []
    if settings['if_ehrs'][combination]:
        feature_cols += EHR_cols
    ## create PE dataFrame
    if settings['dim_pe_ratios'][combination]:
        nonimg = df_ehr[feature_cols].values
        if nonimg.ndim != 2:
            nonimg = nonimg.reshape(1, -1)
        dim_pe = int(len(df_radimoics.columns)*settings['dim_pe_ratios'][combination])
        df_pe = pd.DataFrame(index=df_ehr.index, columns=list(range(dim_pe)))
        nonimg = position_encoding(nonimg, D=dim_pe)
        for i in range(nonimg.shape[0]):
            row = nonimg[i].flatten()
            row = np.pad(row, (dim_pe - len(row), 0), 'edge')
            df_pe.iloc[i] = row
        feature_cols = list(df_pe.columns)
    else:
        df_pe = df_ehr
        
    feature_cols += nodule_cols
    feature_cols += list(df_radimoics.columns)

    ## with radiomics features
    df = pd.DataFrame()
    df = pd.concat([df, df_radimoics], axis=1)
    df = pd.concat([df, df_ehr[nodule_cols]], axis=1)
    if settings['if_ehrs'][combination]:
        df = pd.concat([df, df_pe], axis=1)

    df['time'] = df_ehr['LFU']
    df['event'] = df_ehr['c34_flag'].astype(int)
    df['Lung_Rads_merge'] = df_ehr['Lung_Rads_merge']
    df['label'] = df['event'] & (df['time']<=365)

    df_results['img_name'] = df.index
    df_results.set_index('img_name', inplace=True)
    df_results['event'] = df['event']
    df_results['time'] = df['time']
    df_results['Lung_Rads_merge'] = df['Lung_Rads_merge']
    df_results['label'] = df['label']
    
    return df, df_results, df_ehr, EHR_cols, feature_cols

def make_split(permutation, df, feature_cols, df_results):
    if permutation == 0:
        index_reserved = preprocess_dict.index_reserved_1
    elif permutation == 1:
        index_reserved = preprocess_dict.index_reserved_2
    else:
        index_train, index_val, index_reserved = sampling_index(
            df, val=False, sampling_rate=0.2)
    
    ## spliting train/test set by stratified sampling
    _, _, X_test, _, _, T_test, _, _, E_test = get_sets_from_index(
        df,
        feature_cols, [], [], index_reserved)

    index_train, _, index_val = sampling_index(
        df[~df.index.isin(index_reserved)], 
        val=False, sampling_rate=0.125)

    X_train, X_val, _, T_train, T_val, _, E_train, E_val, _ = get_sets_from_index(
        df,
        feature_cols, index_train, index_val, [])
    
    df_results['split'].loc[index_train] = 'train'
    df_results['split'].loc[index_val] = 'val'
    df_results['split'].loc[index_reserved] = 'test'
    
    return df_results, index_train, index_val, index_reserved, \
        X_train, X_val, X_test, T_train, T_val, T_test, \
            E_train, E_val, E_test
            
def make_model(settings, X_train, T_train, E_train, 
               X_val, T_val, E_val, X_test, T_test, E_test,
               combination_dir_name, df_stats):
    train_unfinished = True
    lr = 2e-5
    while train_unfinished:
        structure = [ {'activation': 'sigmoid', 'num_units': settings['num_units']},  ]
        n_mtlr = MLP_reg_Model(structure=structure, bins=settings['time_bins'])
        model = n_mtlr

        model, error = model.fit(X_train, T_train, E_train, init_method = 'orthogonal',
                optimizer ='rprop', lr = 2e-5, num_epochs = 1000, 
                dropout = 0.1, 
                l2_reg=1e-2, l2_smooth=1e-2, batch_normalization=False, bn_and_dropout=True,
                verbose=True, 
                extra_pct_time = 0.1, 
                is_min_time_zero=True)
        
        if 'gradient exploded' in error:
            lr = lr/2
        elif error == '':
            train_unfinished = True
            break
            
    #### 5 - Cross Validation / Model Performances
    # c_index = mysurv_c_index(model, X_val, T_val, E_val) 
    # if combination_dir_name is not None:
    #     ibs = integrated_brier_score(model, X_val, T_val, E_val, figure_size=(20, 6.5), 
    #             savedir=f'{combination_dir_name}/validation_{round(c_index,2)}.png', show=False)
    c_index = mysurv_c_index(model, X_test, T_test, E_test) 
    # if combination_dir_name is not None:
    #     ibs = integrated_brier_score(model, X_test, T_test, E_test, figure_size=(20, 6.5), 
    #             savedir=f'{combination_dir_name}/test_{round(c_index,2)}.png', show=False)
    # else:
    ibs = integrated_brier_score(model, X_test, T_test, E_test, figure_size=(20, 6.5), 
                                    savedir=None, show=False)
    
    df_stats.loc[combination_dir_name.split('/')[-1], 'c-index'] = c_index
    df_stats.loc[combination_dir_name.split('/')[-1], 'ibs'] = ibs
    return model, df_stats

def get_rad_stats(df, index_reserved, df_stats):
    label = (df['event'].loc[index_reserved]) & (df['time'].loc[index_reserved]<=365)
    report = metrics_utils.prediction_performance_binary(
        label,  df['Lung_Rads_merge'].loc[index_reserved]>=3)
    auc_value, _, _ = metrics_utils.auc_lungrad(label,
        df['Lung_Rads_merge'].loc[index_reserved], full_results=True)
    df_stats.loc['LungRADS', 'AUC'] = auc_value
    df_stats.loc['LungRADS', 'AUC_based-ACC'] = report["accuracy"]
    df_stats.loc['LungRADS', 'AUC_based-FPR'] = report["FPR"]
    df_stats.loc['LungRADS', 'AUC_based-FNR'] = report["FNR"]
    return df_stats

def cal_uq_score(df, index_reserved, X_test, index_train, X_train, model, settings,
                 df_ehr, EHR_cols, feature_cols, df_results, group_num=10):
    df['points'] = df['Lung_Rads_merge']
    uq_reserved = []

    for poi_index in tqdm(index_reserved):
        X_poi = X_test[index_reserved.index(poi_index)]
        ## 2. sort patients and group them according to the sorted similarites
        df_patient_difference = sorted_patients(
            df, poi_index, index_train, X_train, X_poi, model, 
            similarity=settings['similarity'], specified_cols=EHR_cols, 
            df_og=df_ehr, group_num=group_num) #nomogram_num_diff
        
        ## 3. calibrated the group prediction
        ## 4. calculate the similarity with the groups
        errs = calculate_group_errs(
            df, feature_cols, X_poi, df_patient_difference, model, 
            group_num=group_num, alg_calibration='weighted_sum', alg_err='lse')
        start_time = time.perf_counter()
        ## 5. choose a c_index threshold
        c_poi = calculate_c_index(errs, group_num)
        uq_reserved.append(c_poi)
    
    df_results['uq_score'].loc[index_reserved] = uq_reserved
    return df_results, uq_reserved, start_time

def save_prediction(df_results, model, X_train, X_val, X_test,
                    index_train, index_val, index_reserved, settings):
    for X, index in zip([X_train, X_val, X_test], [index_train, index_val, index_reserved]):
        predicted = model.predict_cumulative_hazard(X)
        df_results.loc[index, list(range(settings['time_bins']-1))] = predicted
    return df_results
       
def create_df_uq(index_reserved, uq_reserved, df, model, X_test):
    df_uq = pd.DataFrame(columns=['uq_score', 'event', 'time', 
                                  'hazard_pred_at_time'], index=index_reserved)
    df_uq['uq_score'] = uq_reserved
    df_uq['event'] = df.loc[index_reserved, 'event']
    df_uq['time'] = df.loc[index_reserved, 'time']
    df_uq['hazard_pred_at_time'] = None

    reserved_predicted = model.predict_cumulative_hazard(X_test)
    reserved_predicted[reserved_predicted<0] = 0
    reserved_predicted[reserved_predicted>1] = 1
    x = np.arange(reserved_predicted.shape[0])
    y = cal_time_bin_index(365, model.time_buckets)
    df_uq['hazard_pred_at_time_t'] = reserved_predicted[x, y]
    label = (df['event'].loc[index_reserved]) & (df['time'].loc[index_reserved]<=365)
    df_uq['event_at_time_t'] = label
    
    return df_uq

## choose_pair_by_highest_AUC_lowest_FNR
def choose_pair_by_highest_AUC_lowest_FNR(fnrs_row, fprs_row, accs_row, x):
    fprs_row_ = fprs_row[np.array(x)<=1]
    fnrs_row_ = fnrs_row[np.array(x)<=1]
    accs_row_ = accs_row[np.array(x)<=1]
    try:
        lowest_fnr = min(fnrs_row_)
        fprs_row = fprs_row_[fnrs_row_==lowest_fnr]
        accs_row = accs_row_[fnrs_row_==lowest_fnr]
        fnrs_row = fnrs_row_[fnrs_row_==lowest_fnr]
        
        lowest_fpr = min(fprs_row)
        accs_row = accs_row[fprs_row==lowest_fpr]
        fnrs_row = fnrs_row[fprs_row==lowest_fpr]
        fprs_row = fprs_row[fprs_row==lowest_fpr]
        
    except:
        fprs_row = fprs_row_
        fnrs_row = fnrs_row_
        accs_row = accs_row_

    delta = list(accs_row - fprs_row - fnrs_row)
    recommended_ind = delta.index(max(delta))
    return fprs_row[recommended_ind], fnrs_row[recommended_ind], accs_row[recommended_ind]

## choose_pair_by_Youden
def choose_pair_by_Youden(fnrs_row, fprs_row, accs_row, x):
    fprs_row_ = fprs_row[np.array(x)<=1]
    fnrs_row_ = fnrs_row[np.array(x)<=1]
    accs_row_ = accs_row[np.array(x)<=1]
    try:
        fprs_row = fprs_row_[accs_row_>=0.5]
        fnrs_row = fnrs_row_[accs_row_>=0.5]
        accs_row = accs_row_[accs_row_>=0.5]
    except:
        fprs_row = fprs_row_
        fnrs_row = fnrs_row_
        accs_row = accs_row_

    delta = list(accs_row - fprs_row - fnrs_row)
    recommended_ind = delta.index(max(delta))
    return fprs_row[recommended_ind], fnrs_row[recommended_ind], accs_row[recommended_ind]

def cal_stats(df_uq, index_reserved, df, df_stats, combination, settings):
    ## TTE model at time t
    label = (df['event'].loc[index_reserved]) & (df['time'].loc[index_reserved]<=365)
    y_score = df_uq['hazard_pred_at_time_t'].loc[index_reserved].values
    fpr, tpr, thresholds = roc_curve(label.values, y_score)
    
    ## warning pred
    x = thresholds
    x = pd.unique(x)
    x = sorted(x)

    ## warning uq
    y = sorted(df_uq['uq_score'].unique()) 
    y.insert(0, 0) if y[0]>0 else None
    y.insert(len(y), 1) if y[-1]<1 else None

    X, Y = np.meshgrid(x, y)
    Z = [z for z in map(lambda x,y: cal_acceptable_rate(
        df_uq, x, y, 'event_at_time_t', 'hazard_pred_at_time_t'), X.flatten(), Y.flatten())]
    fprs = np.array([z['FPR'] for z in Z]).reshape(X.shape)
    tprs = np.array([z['TPR'] for z in Z]).reshape(X.shape)

    fprs = np.insert(fprs, fprs.shape[1], 0, axis=1)
    tprs = np.insert(tprs, tprs.shape[1], 0, axis=1)

    auc_values = [auc(fprs[i], tprs[i]) for i in range(fprs.shape[0])]
    max_ind = auc_values.index(max(auc_values))
    df_stats.loc[settings['dir_names'][combination], 'AUC'] = auc_values[max_ind]
    
    accs_matrix = np.array([z['accuracy'] for z in Z]).reshape(X.shape)
    fprs_matrix = np.array([z['FPR'] for z in Z]).reshape(X.shape)
    fnrs_matrix = np.array([z['FNR'] for z in Z]).reshape(X.shape)
    
    fnrs_row = fnrs_matrix[max_ind]
    fprs_row = fprs_matrix[max_ind]
    accs_row = accs_matrix[max_ind]
    
    p, n, a = choose_pair_by_highest_AUC_lowest_FNR(fnrs_row, fprs_row, accs_row, x)
    df_stats.loc[settings['dir_names'][combination], 'AUC_based-FPR'] = p
    df_stats.loc[settings['dir_names'][combination], 'AUC_based-FNR'] = n
    df_stats.loc[settings['dir_names'][combination], 'AUC_based-ACC'] = a
    
    p, n, a = choose_pair_by_Youden(fnrs_row, fprs_row, accs_row, x)
    df_stats.loc[settings['dir_names'][combination], 'Youden_based-FPR'] = p
    df_stats.loc[settings['dir_names'][combination], 'Youden_based-FNR'] = n
    df_stats.loc[settings['dir_names'][combination], 'Youden_based-ACC'] = a
    
    return df_stats

