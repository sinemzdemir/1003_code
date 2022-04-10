"""
This module trains protein function models and reveals best model and hyperparameters. The module structure is the following:
- The module implements ``check_for_at_least_two_class_sample_exits`` method. The method takes a dataframes as input.
The input dataframe has varying number of columns. Each column represent a class (i.e. GO ids). 
The methods analyze the data frame to control at least two positive sample exits for each class.

- The module implements ``create_valid_kfold_object_for_multilabel_splits`` method. The method takes dataframe of labels, 
samples and kfold object as input. The methods returns a valid kfold object which splits train/test folds 
which includes at least 2 folds with positive samples for each class.

- The module implements `report_results_of_training`` method. The method takes dictionary of results as input and 
writes down results to files in a proper format .

- The module implements ``select_best_model_with_hyperparameter_tuning`` method. The method a dataframe and 
list of preferred model names as input. The dataframe has 3 columns 'Label','Entry' and 'Vector'. The method 
trains models and search for best model and hyperparameters.

- The module implements ``train_model_with_datasets`` method. The method takes a dataframes as input.
The function takes dataset directory as input, initialize relevant parameters, calls training and reporting functions.
It is basicly acts as an API for this module.
"""
#import tensorflow as tf
import ast
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from random import random
from sklearn.metrics import multilabel_confusion_matrix
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from HoloProtRepAFPML import evaluate
from sklearn.metrics import make_scorer
from sklearn import metrics
from HoloProtRepAFPML import pytorch_network
from HoloProtRepAFPML.pytorch_network import NN
import torch

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
std_rc_ma_ce_cv = []
std_hamm_cv = []
mc = []
acc = []
y_proba = []


def intersection(real_annot, pred_annot):
    count=0
    tn=0
    tp=0
    for i in range(len(real_annot)):
        if(real_annot[i]==pred_annot[i]):
            if(real_annot[i]==0):
                tn+=1
            else:
                tp+=1
            count+=1
                
    return tn,tp

def evaluate_annotation_f_max(real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    tn=0
    tp=0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        
        tn,tp=intersection(real_annots[i], pred_annots[i])
        fp = len(pred_annots[i]) - tp
        fn = len(real_annots[i]) - tn
        total += 1
        recall = tp /(1.0 * (tp + fn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tp / (1.0 * (tp + fp))
            p += precision
    
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    
    return f
    
def check_for_at_least_two_class_sample_exits(y):
    for column in list(y):
        column_sum = np.sum(y[column])
        if column_sum < 2:
          print('At least 2 positive samples needed for each class {0} class has {1} positive samples'.format(column,column_sum))
          return False
    return True

def create_valid_kfold_object_for_multilabel_splits(X,y,kf):
    check_for_at_least_two_class_sample_exits(y)
    y_df=pd.DataFrame(y)
    sample_class_occurance = dict(zip(y_df,np.zeros(len(y_df))))
    for column in y_df:
        for fold_train_index,fold_test_index in kf.split(X,y_df):
            fold_col_sum = np.sum(y_df.iloc[fold_test_index,:][column].array)
            if fold_col_sum > 0:
                sample_class_occurance[column] += 1 

    for key in sample_class_occurance:
        value = sample_class_occurance[key]
        if value < 2:
            random_state = np.random.randint(1000)
            print("Random state changed since at least two positive samples are needed in different train/test folds.\
                    \nHowever, only one fold exits with positive samples for class {0}".format(key))
            print("Selected random state is {0}".format(random_state))
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            create_valid_kfold_object_for_multilabel_splits(X,y_df,kf)
        else:
            return kf


def select_best_model_with_hyperparameter_tuning(representation_name,name_of_go_id_with_protein_names,integrated_lst,models=["SVC","RandomForestClassifier","KNeighborsClassifier","Neural_Network"],auto=True,total_run_count=1):
     
    '''label_list = list(integrated_dataframe['Label'])   
    mlt = MultiLabelBinarizer()
    labels=[tuple(arr) for arr in label_list]
    model_label = mlt.fit_transform(labels)
    low=[list(i).count(1) for i in model_label]
    drop_lst=[]
    for i in range(len(low)):
        if low[i]<2:
            drop_lst.append(i)
    integrated_dataframe=integrated_dataframe.drop(drop_lst,axis=0)    '''
    integrated_dataframe=pd.DataFrame(integrated_lst[0])
    label_list=[label for label in integrated_dataframe['Label']]
    #label_list = [ast.literal_eval(label) for label in integrated_dataframe['Label']]  
    protein_representation = integrated_dataframe.drop(["Label", "Entry"], axis=1)
    proteins=list(integrated_dataframe['Entry'])
    vectors=list(protein_representation['Vector'])
    protein_and_representation_dictionary=dict(zip(proteins,vectors ))
    row = protein_representation.shape[0]
    row_val = round(math.sqrt(row), 0)
    mlt = MultiLabelBinarizer()
    model_label = mlt.fit_transform(label_list)   
    protein_representation_array = np.array(list(protein_representation['Vector']), dtype=float)
    model_label_array = np.array(model_label)
    predictions_list = []
    best_parameter_df = pd.DataFrame(
    columns={"representation_name", 'classifier_name',  'best parameter'})
    result_dict={}
    classifier_name_lst=[]
    index=0
    model_count=0
    file_name = "_"    
    path=os.path.dirname(os.getcwd())+'/results'
    if 'training'  not in os.listdir(path):
        os.makedirs(path+"/training",exist_ok=True)        
    file_name=file_name.join(models)
    best_param_list=[]
    for classifier in models:            
        index+=1
        m=0
        model_label_pred_lst=[]
        label_lst=[]
        protein_name=[]
        if(classifier== "Neural_Network"):
            input_size= len(protein_representation_array[0])
            for i in model_label:
                if len(i)>m :
                    class_number= len(i)                                   
       
        else: 
            if classifier== "RandomForestClassifier":
                classifier_ = RandomForestClassifier(random_state=42)
                classifier_name = type(classifier_).__name__
                classifier_multilabel = OneVsRestClassifier(classifier_, n_jobs=-1)
                model_pipline = Pipeline([('scaler', StandardScaler()), ('model_classifier', classifier_multilabel)])
                    
                parameters = {'model_classifier__estimator__n_estimators': [10, 100],
                          'model_classifier__estimator__max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11],
                          'model_classifier__estimator__min_samples_leaf': [1, 5, 10, 20, 100]}

            elif classifier == "SVC":
                classifier_ = SVC(probability=True, random_state=total_run_count)
                classifier_name = type(classifier_).__name__
                classifier_multilabel = OneVsRestClassifier(classifier_, n_jobs=-1)
                model_pipline = Pipeline([('scaler', StandardScaler()), ('model_classifier', classifier_multilabel)])
                    
                parameters = {
                'model_classifier__estimator__C': [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1, 2, 4, 8, 16, 32, 64,
                                                   128, 256, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14,
                                                   2 ** 15],
                'model_classifier__estimator__gamma': [2 ** -15, 2 ** -14, 2 ** -13, 2 ** -12, 2 ** -11, 2 ** -10,
                                                       2 ** -9, 2 ** -8, 2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3,
                                                       2 ** -2, 2 ** -1, 1, 2, 4, 8],
                'model_classifier__estimator__kernel': ['linear', 'poly', 'rbf'],
                'model_classifier__estimator__max_iter': [-1, 5, 10, 10, 20, 30, 40, 50, 100]}
               

            elif classifier == "KNeighborsClassifier":
                classifier_ = KNeighborsClassifier()
                classifier_name = type(classifier_).__name__
                up_limit = int(math.sqrt(int(len(model_label) / 5)))
                classifier_multilabel = classifier_
                model_pipline = Pipeline([('scaler', StandardScaler()), ('model_classifier', classifier_multilabel)])
                    
                k_range = list(range(1, up_limit))

                parameters = {'model_classifier__n_neighbors': k_range,
                          'model_classifier__weights': ["uniform", "distance"],
                          'model_classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                          'model_classifier__leaf_size': list(range(1, int(len(model_label) / 5))), 'model_classifier__p': [1, 2]}
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)        
        kf=create_valid_kfold_object_for_multilabel_splits(protein_representation,model_label,kf)
        if(classifier== "Neural_Network"):
            model_count=model_count+1
            classifier_name_lst.append("Neural_Network")
            classifier_name="Neural_Network"
            
            
        else:
            model_count=model_count+1                                
          
          
                
        if (classifier== "Neural_Network"):
            f_max_cv = []          
            for fold_train_index, fold_test_index in kf.split(protein_representation, model_label):
                for i in model_label[fold_test_index]:
                    if len(i)>m :
                        class_number= len(i)
            
                protein_representation_fold=pd.DataFrame(protein_representation['Vector'],index=list(fold_test_index))
                model_label_pred,parameter,model = NN(protein_representation_fold,model_label[fold_test_index],input_size,class_number,representation_name)
                
                model_label_pred_lst.append(model_label_pred.detach().numpy())
                label_lst.append(model_label[fold_test_index])
                for vec in protein_representation_array[fold_test_index]:
                    for protein, vector in protein_and_representation_dictionary.items():  
                        if str(vector) == str(list(vec)):
                            protein_name.append(protein)
                            continue
                fmax = 0.0
                tmax = 0.0
                for t in range(1, 101):
                    threshold = t / 100.0
                    fscore=evaluate_annotation_f_max( model_label[fold_test_index],model_label_pred )
                    if fmax < fscore:
                        fmax = fscore
                        tmax = threshold
                f_max_cv.append(fmax)
                
                #rep_name="_".join(representation_name)
                paths=path+'/'+'training'+'/'+representation_name+'_'+classifier_name+'_'+name_of_go_id_with_protein_names+".pt"
                torch.save(model,paths )
            if auto==True:
                best_param_list.append(parameter) 
           
            rep_name='_'.join(representation_name)
            
            
            pd.DataFrame(parameter).to_csv(path+'/'+'training'+'/'+representation_name+'_'+name_of_go_id_with_protein_names+'_'+'neural_network'+'_best_parameter' +'.csv',index=False)
            evaluate.evaluate(kf, mlt, protein_representation, model_label_pred_lst, label_lst, classifier_name,representation_name,name_of_go_id_with_protein_names,protein_and_representation_dictionary,f_max_cv,file_name,index,"training")
            
            
        else:
                    
            f_max_cv = []
            model_label_pred = cross_val_predict(model_tunning.best_estimator_, protein_representation_array,model_label, cv=kf, n_jobs=-1)  
            for fold_train_index, fold_test_index in kf.split(protein_representation, model_label):
               
                model_label_pred= model_tunning.best_estimator_.predict(protein_representation_array[fold_test_index])
                model_label_pred_lst.append(model_label_pred)
                label_lst.append(model_label[fold_test_index])
                for vec in protein_representation_array[fold_test_index]:
                    for protein, vector in protein_and_representation_dictionary.items():  
                        if str(vector) == str(list(vec)):
                            protein_name.append(protein)
                            continue
       
            fmax = 0.0
            tmax = 0.0
            for t in range(1, 101):
                threshold = t / 100.0
                fscore=evaluate_annotation_f_max(model_label[fold_test_index],model_label_pred )
                if fmax < fscore:
                    fmax = fscore
                    tmax = threshold
            f_max_cv.append(fmax)
       
            evaluate.evaluate(kf, "multilabel", protein_representation, model_label_pred_lst, label_lst, classifier_name,representation_name,name_of_go_id_with_protein_names,protein_and_representation_dictionary,f_max_cv,file_name,index,"training")
        col_names=mlt.classes_
        label_predictions=pd.DataFrame(np.concatenate(model_label_pred_lst),columns=col_names)
        ls=np.concatenate(model_label_pred_lst)
        label_predictions.insert(0, "protein_id", protein_name)
        label_predictions["prediction_values"]=[np.array(i, dtype=np.int).tolist() for i in ls]      
        
        label_predictions.to_csv(path+'/'+'training'+'/'+representation_name +'_'+name_of_go_id_with_protein_names+ '_' + classifier_name +'_training'+ "_predictions5cv.tsv",sep="\t", index=False)
    if auto==True:
        return best_param_list
    else:
        return None
            


'''data_path="/media/DATA/home/sinem/tekli_datalar/biological_process_data_combinations/biological_process_ksep_dataframe_Low_Shallow.pkl"
pkl_file = open(data_path, 'rb')
readed_dataset = pickle.load(pkl_file)
pkl_file.close()
representation_name="ksep"
dataset_dir=[]
model=[RandomForestClassifier]  #,"Neural_Network",
dataset_dir.append(readed_dataset)
result_list=[]
mean_result_list=[]
classifier_name_lst=[]
data_len=0
for integrated_dataframe in dataset_dir:
    
    integrated_dataframe.columns=['Entry', 'Label', 'Aspect', 'Vector']
    integrated_dataframe=integrated_dataframe.drop(['Aspect'],axis=1)
    data_len=len(integrated_dataframe['Vector'][0])
    select_best_model_with_hyperparameter_tuning(representation_name, integrated_dataframe,model)'''
    

    


