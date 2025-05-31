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
from permutation_run_utils import make_df_stats, make_df, make_split,\
    make_model, get_rad_stats, cal_uq_score, create_df_uq, cal_stats, \
        save_prediction

from decimal import Decimal

settings = {}
settings['dim_pe_ratios'] = [0, 0, 0.5, 1.0, 2.0]
settings['if_ehrs'] = [False, True, True, True, True]
settings['dir_names'] = ['radiomics_only', 'PE_0', 'PE_0d5', 'PE1_0', 'PE_2d0']
settings['similarity'] = 'nomogram_num_diff'

settings['time_bins'] = 100
settings['num_units'] = 150

def create_display_number(xx):
    if xx < 1e-2:
        return '%.2E'%Decimal(xx)
    else:
        return str(round(xx, 2))


def run_permutation(permutation, combinations=list(range(5))):
    permutation_dir = f"results/{permutation}/"
    df_stats = make_df_stats(permutation_dir)
    
    for combination in combinations:
        combination_dir_name = f"{permutation_dir}{settings['dir_names'][combination]}" 
        os.makedirs(combination_dir_name, exist_ok=True)
        result_dir = f'{combination_dir_name}/results.csv'
        ####### create dataset ####### 
        df, df_results, df_ehr, EHR_cols, feature_cols = make_df(
            result_dir, settings, combination) 
        ######## split dataset ####### 
        df_results, index_train, index_val, index_reserved, \
            X_train, X_val, X_test, T_train, T_val, T_test, \
            E_train, E_val, E_test = make_split(
                permutation, df, feature_cols, df_results)
        ######## train model #######
        model, df_stats = make_model(settings, X_train, T_train, E_train, 
               X_val, T_val, E_val, X_test, T_test, E_test,
               combination_dir_name, df_stats)
        df_results = save_prediction(df_results, model, X_train, X_val, X_test,
                    index_train, index_val, index_reserved, settings)
        ########## UQ ############
        df_results, uq_reserved = cal_uq_score(df, index_reserved, X_test, 
                index_train, X_train, model, settings, df_ehr, EHR_cols, 
                feature_cols, df_results)
        df_uq = create_df_uq(index_reserved, uq_reserved, df, model, X_test)
        df_stats= cal_stats(df_uq, index_reserved, df, df_stats, combination, settings)
        
        df_uq.to_csv(f'{combination_dir_name}/df_uq.csv')
        df_results.to_csv(result_dir)
    
    df_stats = get_rad_stats(df, index_reserved, df_stats)
    df_stats.to_csv(f'{permutation_dir}stats.csv')
    return df_stats
        
        
if __name__ == '__main__':
    for i in tqdm(range(1000)):
        run_permutation(i)
    