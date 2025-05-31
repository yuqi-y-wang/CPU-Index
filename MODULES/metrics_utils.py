import sklearn
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def prediction_performance_binary(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.flatten()
    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    # # Recall or true positive rate
    TPR = TP/(TP+FN) 
    TNR = TN/(TN+FP)
    report['TPR'] = TPR
    report['TNR'] = TNR
    # # Precision or positive predictive value
    # PPV = TP/(TP+FP)
    # NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    # # False discovery rate
    # FDR = FP/(TP+FP)
    # # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN)
    report['FPR'] = FPR
    report['FNR'] = FNR
    report['confusion_matrix'] = cm
    report['TP'] = TP
    report['FP'] = FP
    return report

## function: calculate the auc from lungrads
def auc_lungrad(label, radscore, full_results=False):
    y_score = np.zeros_like(radscore, dtype=float)
    score_map = [0.17, 0.33, 0.5, 0.625, 0.75, 0.875]
    for i in range(1, 7):
        y_score[radscore==i] = score_map[i-1]
    if full_results:
        return roc_auc_score(label, y_score), label, y_score
    return roc_auc_score(label, y_score)



# ## function: running stratified CV for a claasifier
# def cross_val_score_stratified(clf, X_train, X_test, y_train, y_test):
#     ## get CV folds
#     skf = StratifiedKFold(n_splits=5, shuffle=True)
#     for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
#         print(f'Fold {i}:')
#         X_train_cv = X_train[train_index]
#         y_train_cv = y_train[train_index]
#         X_val_cv = X_train[val_index]
#         y_val_cv = y_train[val_index]
#         clf.fit(X_train_cv, y_train_cv)

#         y_pred = clf.predict(X_val_cv)
#         y_score = clf.predict_proba(X_val_cv)[:,1]
#         report = metrics_utils.prediction_performance_binary(y_val_cv, y_pred)
#         auc = roc_auc_score(y_val_cv, y_score)
#         print('Val: ACC {}, FPR {}, FNR {}, auc {}'.format(report['accuracy'], report['FPR'], report['FNR'], auc))
#         radscore = df[label_cols[0]][ind_train][val_index]
#         report_radiologist = metrics_utils.prediction_performance_binary(y_val_cv, radscore>=3)
#         print('Radiologist: ACC {}, FPR {}, FNR {}, AUC {}'.format(
#             report_radiologist['accuracy'], report_radiologist['FPR'], report_radiologist['FNR'], metrics_utils.auc_lungrad(y_val_cv, radscore)))


#         y_pred = clf.predict(X_test)
#         y_score = clf.predict_proba(X_test)[:,1]
#         report = metrics_utils.prediction_performance_binary(y_test, y_pred)
#         auc = roc_auc_score(y_test, y_score)
#         print('Test: ACC {}, FPR {}, FNR {}, auc {}'.format(report['accuracy'], report['FPR'], report['FNR'], auc))
#         radscore = df[label_cols[0]][ind_test]
#         report_radiologist = metrics_utils.prediction_performance_binary(y_test, radscore>=3)
#         print('Radiologist: ACC {}, FPR {}, FNR {}, AUC {}'.format(
#             report_radiologist['accuracy'], report_radiologist['FPR'], report_radiologist['FNR'], metrics_utils.auc_lungrad(y_test, radscore)))