import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil
#from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef
import pickle
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from random import random
from sklearn.metrics import multilabel_confusion_matrix
import sys
from sklearn.svm import SVC
'''from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.preprocessing import StandardScaler'''
from sklearn.neighbors import KNeighborsClassifier
#from imblearn.pipeline import Pipeline
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from HoloProtRepAFPML import binary_pytorch_network

#from  pytorch_network import NN

from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from HoloProtRepAFPML import report_results_of_training


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc
    

def evaluate(kf, protein_representation,model_label_pred_lst, label_lst, classifier_name,representation_name,protein_and_representation_dictionary,file_name,index,classifier_type,mlt="binary"):


    acc_cv = []
    f1_mi_cv = []
    f1_ma_cv = []
    f1_we_cv = []
    pr_mi_cv = []
    pr_ma_cv = []
    pr_we_cv = []
    rc_mi_cv = []
    rc_ma_cv = []
    rc_we_cv = []
    hamm_cv = []
    mcc_cv = []
    std_acc_cv = []
    std_f1_mi_cv = []
    std_f1_ma_cv = []
    std_f1_we_cv = []
    std_pr_mi_cv = []
    std_pr_ma_cv = []
    std_pr_we_cv = []
    std_rc_mi_cv = []
    std_rc_ma_cv = []
    std_rc_we_cv = []
    std_hamm_cv = []
    
    GO_ids=[]   

    protein_representation_array = np.array(list(protein_representation['Vector']), dtype=float)          
      
    protein_name=[]    
    mc = []
    acc = []
    tn_lst = []
    fn_lst = []
    tp_lst = []

    fp_lst = []       
    mean_result_list = []    
    result_list = []      
    auc_cv=[]
    std_auc_cv=[]
    fp_lst = []
    total = 0
    y_lst = []

    y_class = []
    predictions_list = []
    result_list = []   
    y_test_list = []
    auc_cv=[]
    std_auc_cv=[]
   
    for fold_index in range(len(model_label_pred_lst)):
  
        tn, fp, fn, tp = confusion_matrix(label_lst[fold_index], model_label_pred_lst[fold_index]).ravel()         
        tn_lst.append(tn)
        tp_lst.append(tp)
        fn_lst.append(fn)
        fp_lst.append(fp)
        acc.append((tp+tn)/(tp+fn+fp+tn))
        mc.append(((tp * tn) - (fp * fn)) / math.sqrt(
                (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc_cv.append(mc)
        ac = np.mean(acc)
        acc_cv.append(ac)
        f1_mi = f1_score(label_lst[fold_index], model_label_pred_lst[fold_index], average="micro")
        f1_mi_cv.append(f1_mi)
        f1_ma = f1_score(label_lst[fold_index], model_label_pred_lst[fold_index], average="macro")
        f1_ma_cv.append(f1_ma)
        f1_we = f1_score(label_lst[fold_index], model_label_pred_lst[fold_index], average="weighted")
        f1_we_cv.append(f1_we)
        pr_mi = precision_score(label_lst[fold_index], model_label_pred_lst[fold_index], average="micro")
        pr_mi_cv.append(pr_mi)
        pr_ma = precision_score(label_lst[fold_index], model_label_pred_lst[fold_index], average="macro")
        pr_ma_cv.append(pr_ma)
        pr_we = precision_score(label_lst[fold_index], model_label_pred_lst[fold_index], average="weighted")
        pr_we_cv.append(pr_we)
        rc_mi = recall_score(label_lst[fold_index], model_label_pred_lst[fold_index], average="micro")
        rc_mi_cv.append(rc_mi)
        rc_ma = recall_score(label_lst[fold_index], model_label_pred_lst[fold_index], average="macro")
        rc_ma_cv.append(rc_ma)
        rc_we = recall_score(label_lst[fold_index], model_label_pred_lst[fold_index], average="weighted")
        rc_we_cv.append(rc_we)
        hamm = hamming_loss(label_lst[fold_index], model_label_pred_lst[fold_index])
        hamm_cv.append(hamm)
        auc = compute_roc(label_lst[fold_index], model_label_pred_lst[fold_index])
        auc_cv.append(auc)
    std_acc_cv.append(np.std(acc_cv))
    std_f1_mi_cv.append(np.std(f1_mi_cv))
    std_f1_ma_cv.append(np.std(f1_ma_cv))
    std_f1_we_cv.append(np.std(f1_we_cv))
    std_pr_mi_cv.append(np.std(pr_mi_cv))
    std_pr_ma_cv.append(np.std(pr_ma_cv))
    std_pr_we_cv.append(np.std(pr_we_cv))
    std_rc_mi_cv.append(np.std(rc_mi_cv))
    std_rc_ma_cv.append(np.std(rc_ma_cv))
    std_rc_we_cv.append(np.std(rc_we_cv))
    std_hamm_cv.append(np.std(hamm_cv))
    std_auc_cv.append(np.std(auc_cv))
    
    result_list.append(
            [representation_name, classifier_name, acc_cv, std_acc_cv, f1_mi_cv,
             std_f1_mi_cv, f1_ma_cv, std_f1_ma_cv, f1_we_cv, std_f1_we_cv, pr_mi_cv, std_pr_mi_cv, pr_ma_cv,
             std_pr_ma_cv, pr_we_cv, std_pr_we_cv, rc_mi_cv, std_rc_mi_cv, rc_ma_cv, std_rc_ma_cv, rc_we_cv,
             std_rc_we_cv, hamm_cv, std_hamm_cv,auc_cv,std_auc_cv, mcc_cv])
    mean_result_list.append(
            [representation_name, classifier_name,np.mean(acc_cv), np.mean(std_acc_cv), 
                                                      np.mean(f1_mi_cv),
                                                      np.mean(std_f1_mi_cv),
                                                      np.mean(f1_ma_cv),
                                                      np.mean(std_f1_ma_cv),
                                                      np.mean(f1_we_cv),
                                                      np.mean(std_f1_we_cv),
                                                      np.mean(pr_mi_cv),
                                                      np.mean(std_pr_mi_cv),
                                                      np.mean(pr_ma_cv),
                                                      np.mean(std_pr_ma_cv),
                                                      np.mean(pr_we_cv),
                                                      np.mean(std_pr_we_cv),
                                                      np.mean(rc_mi_cv),
                                                      np.mean(std_rc_mi_cv),
                                                      np.mean(rc_ma_cv),
                                                      np.mean(std_rc_ma_cv),
                                                      np.mean(rc_we_cv),
                                                      np.mean(std_rc_we_cv),
                                                      np.mean(hamm_cv),
                                                      np.mean(std_hamm_cv),np.mean(auc_cv),np.mean(std_auc_cv),
                                                      np.mean(mcc_cv)])



    report_results_of_training.report_results_of_training(representation_name,result_list,mean_result_list,classifier_name,file_name,index,classifier_type)
    
    
    return (result_list, mean_result_list)


