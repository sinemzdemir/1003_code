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
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import evaluate
from sklearn.metrics import make_scorer
from sklearn import metrics
import pandas as pd
from  pytorch_network import NN
import ast
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

def report_results_of_training(representation_name,result_list,mean_result_list,index,models,classifier_names,classifier_name_lst):
    classifier_len=len(models)
    file_name = "_" 
    file_name.join(classifier_name_lst)         
    if (index==1):    
        result_df = pd.DataFrame(columns=["representation_name","classifier_name", "acc", "std_acc","f_max","std_f_max",  "f1_mi", "std_f1_mi", "f1_ma", "std_f1_ma", "f1_we", "std_f1_we", "pr_mi", "std_pr_mi","pr_ma", "std_pr_ma", "pr_we", "std_pr_we", "rc_mi", "std_rc_mi","rc_ma", "std_rc_ma", "rc_we", "std_rc_we",  "hamm", "std_hamm","auc","std_auc","mcc"])           
        result_df.loc[classifier_len] = result_list[0]
        result_df.reset_index(drop=True, inplace=True)                
        mean_result_dataframe = pd.DataFrame(columns=["representation_name","classifier_name", "acc", "std_acc", "f_max","std_f_max", "f1_mi", "std_f1_mi", "f1_ma", "std_f1_ma", "f1_we", "std_f1_we", "pr_mi", "std_pr_mi","pr_ma", "std_pr_ma", "pr_we", "std_pr_we", "rc_mi", "std_rc_mi","rc_ma", "std_rc_ma", "rc_we", "std_rc_we",  "hamm", "std_hamm","auc","std_auc","mcc"])
           
        mean_result_dataframe.loc[classifier_len] = mean_result_list[0]
        mean_result_dataframe.reset_index(drop=True, inplace=True)
        path = os.getcwd()    
                    
        if "results" in os.listdir(path):
            os.chdir(path+'/results')  
            result_df.to_csv(representation_name+file_name+".tsv", sep="\t", index=False)
            mean_result_dataframe.to_csv(representation_name+file_name+'_means'+".tsv", sep="\t", index=False)
        else:
            os.makedirs(path+"/results",exist_ok=True)
            os.chdir(path+'/results')         
            result_df.to_csv(representation_name+file_name+".tsv", sep="\t", index=False)
            mean_result_dataframe.to_csv(representation_name+file_name+'_means'+".tsv", sep="\t", index=False)
                
    elif (index>1):
        path = os.getcwd()
        os.chdir(path+'/results')
        result_df=pd.read_csv(representation_name+file_name+".tsv", sep="\t")
        result_df.loc[classifier_len*data_len] = result_list[0]
        result_df.reset_index(drop=True, inplace=True)
        mean_result_dataframe=pd.read_csv(representation_name+file_name+'_means'+".tsv", sep="\t")
        mean_result_dataframe.loc[classifier_len] = mean_result_list[0]
        mean_result_dataframe.reset_index(drop=True, inplace=True)
        path_working_directory = os.getcwd()    
        path=os.path.dirname(os.getcwd())
        os.chdir(path) 
    pass



def AutomatedFunctionPrediction(data_path, integrated_dataframe,parameteter_file):
    parameters=pd.read_csv(parameteter_file)
    label_list = list(integrated_dataframe['Label'])
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
           
    for i in range(len(parameters)):
        if (parameters['classifier_name'][i]=='RandomForestClassifier'):
            classifier_name='RandomForestClassifier'            
            params=ast.literal_eval(parameters['best parameter'][i])
            classifier=RandomForestClassifier(max_depth=params['model_classifier__estimator__max_depth'],min_samples_leaf=params['model_classifier__estimator__min_samples_leaf'],n_estimators=params['model_classifier__estimator__n_estimators'])
            classifier_multilabel = OneVsRestClassifier(classifier, n_jobs=-1)
            model_pipline = Pipeline([('scaler', StandardScaler()), ('model_classifier', classifier_multilabel)])
        elif (parameters['classifier_name'][i]=='SVC'):
            classifier_name='SVC'            
            params=ast.literal_eval(parameters['best parameter'][i])
            classifier=SVC(C=params['model_classifier__estimator__C'],gamma=params['model_classifier__estimator__gamma'],kernel=params['model_classifier__estimator__kernel'],max_iter=params['model_classifier__estimator__max_iter'])
            classifier_multilabel = OneVsRestClassifier(classifier, n_jobs=-1)
            model_pipline = Pipeline([('scaler', StandardScaler()), ('model_classifier', classifier_multilabel)])    
        elif (parameters['classifier_name'][i]=='KNeighborsClassifier'):
            classifier_name='KNeighborsClassifier'
            params=ast.literal_eval(parameters['best parameter'][i])
            up_limit = int(math.sqrt(int(len(model_label) / 5)))
            k_range = list(range(1, up_limit))
            classifier=KNeighborsClassifier(n_neighbors=params['model_classifier__n_neighbors'],weights=params['model_classifier__weights'],algorithm=params['model_classifier__algorithm'],leaf_size=params['model_classifier__leaf_size'])
            model_pipline = Pipeline([('scaler', StandardScaler()), ('model_classifier', classifier)])
          
        elif (parameters['classifier_name'][i]=='Neural_Network'):
            m=0
            classifier_name='Neural_Network'          
            input_size= len(protein_representation_array[0])
            for ii in model_label:
                if len(ii)>m :
                    class_number= len(ii)      
                                                 
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf=create_valid_kfold_object_for_multilabel_splits(protein_representation,model_label,kf)
    #model_pipline.fit(protein_representation_array, model_label)            
    
        
    if  (classifier== "Neural_Network"):
        result_list,mean_result_list,parameter=evaluate.evaluate_nn(kf,mlt,protein_representation, label_list,model_label,classifier_name,representation_name,protein_and_representation_dictionary)
        report_results_of_training(representation_name,result_list,mean_result_list,index,models,classifier_name,classifier_name_lst)
    
              
    else:
        path=os.path.dirname(os.path.dirname(os.getcwd()))
        os.chdir(path)
        result_list,mean_result_list=evaluate.evaluate(kf,mlt,model_pipline['model_classifier'],protein_representation, label_list,model_label,classifier_name,representation_name,protein_and_representation_dictionary)
               
        evaluate(kf, mlt,model_pipline['model_classifier'].__class__.__name__ , protein_representation, label_list, model_label, classifier_name,representation_name,protein_and_representation_dictionary)
        report_results_of_training(representation_name,result_list,mean_result_list,index,models,classifier_name,classifier_name_lst)  
            
                
    
    
parameteter_file="/media/DATA/home/sinem/holoprotrep/results/results/ksep_best_parameter.csv"
data_path="/media/DATA/home/sinem/tekli_datalar/biological_process_data_combinations/biological_process_ksep_dataframe_Low_Shallow.pkl"
pkl_file = open(data_path, 'rb')
readed_dataset = pickle.load(pkl_file)
pkl_file.close()
representation_name="ksep"
dataset_dir=[]
model=[KNeighborsClassifier]  #"Neural_Network",
dataset_dir.append(readed_dataset)
result_list=[]
mean_result_list=[]
classifier_name_lst=[]
data_len=0
for integrated_dataframe in dataset_dir:
    
    integrated_dataframe.columns=['Entry', 'Label', 'Aspect', 'Vector']
    integrated_dataframe=integrated_dataframe.drop(['Aspect'],axis=1)
    data_len=len(integrated_dataframe['Vector'][0])
    AutomatedFunctionPrediction(data_path, integrated_dataframe,parameteter_file)
    