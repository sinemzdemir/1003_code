import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil
import joblib
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
import ast
from sklearn.svm import SVC
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from HoloProtRepAFPML import binary_evaluate
import joblib
from sklearn.metrics import make_scorer
from sklearn import metrics
from HoloProtRepAFPML import AutomatedFunctionPredictionML
from HoloProtRepAFPML import binary_pytorch_network
#from  HoloProtRepAFPML.pytorch_network import NN
from sklearn.preprocessing import StandardScaler
import copy
def make_prediction(representation_name,data_preproceed,tested_model,classifier_name):

     
    protein_representation = data_preproceed.drop(["Entry"], axis=1)
    proteins=list(data_preproceed['Entry'])
    vectors=list(protein_representation['Vector'])
    protein_and_representation_dictionary=dict(zip(proteins,vectors ))
    row = protein_representation.shape[0]
    row_val = round(math.sqrt(row), 0)
    label_list = [ast.literal_eval(label) for label in protein_representation['Vector']]  
    protein_representation_array = np.array(label_list, dtype=float)    
    f_max_cv = []  
    path=os.path.dirname(os.getcwd())+'/results'
    if 'prediction'  not in os.listdir(path):
        os.makedirs(path+"/prediction",exist_ok=True)         
    
    index=0
    
    for i in range(len(classifier_name)):
        label_lst=[]
        model_label_pred_lst=[]
        if (classifier_name[i]=='RandomForestClassifier'):
           
            #params=ast.literal_eval(parameters[i]['best parameter'][0])
            model = joblib.load(tested_model[i])
            sc = StandardScaler()
            label_list_std = sc.fit_transform(label_list)           
            model_label_pred_lst=model.predict(label_list_std)   
            index=index+1
        elif (classifier_name[i]=='SVC'):
                    
            #params=ast.literal_eval(parameters[i]['best parameter'][0])
            model = joblib.load(tested_model[i])
            
            model_label_pred_lst=model.predict(label_list)   
        elif (classifier_name[i]=='KNeighborsClassifier'):
                         
            #params=ast.literal_eval(parameters[i]['best parameter'][0])
            model = joblib.load(tested_model[i])           
            model_label_pred_lst=model.predict(label_list)   
      
     
        if (classifier_name[i]== "Fully Connected Neural Network"):
            
           
            input_size= len(protein_representation_array[0])
            class_num=1
            model_class=binary_pytorch_network.model_call(input_size,class_num)
            model_class.load_state_dict(copy.deepcopy(torch.load(tested_model[i])))
            model_class.eval()
            x = torch.tensor(label_list)
            x=x.double()
            model_label_pred_lst=model_class(x) 
            model_label_pred_lst[model_label_pred_lst >= 0.] = 1    
            model_label_pred_lst[model_label_pred_lst < 0.] = 0
            input_size= len(protein_representation_array)
            protein_name=[]              
            for protein, vector in protein_and_representation_dictionary.items():                 
                protein_name.append(protein)
            model_label_pred_lst= model_label_pred_lst.type(torch.int16)
            model_label_pred_lst=[k for i in model_label_pred_lst.tolist() for k in i]    
            
            
            # Get training and test loss histories
           
                       
        else:       
            protein_name=[]              
            for protein, vector in protein_and_representation_dictionary.items():                 
                protein_name.append(protein)
           
        col_names=["Label"]
                  
        label_predictions=pd.DataFrame(model_label_pred_lst,columns=col_names)
    
        label_predictions.insert(0, "protein_id", protein_name)                  
        label_predictions.to_csv(path+'/prediction/'+representation_name[0]+'_'+"prediction_" +"binary_classifier"+ '_' + classifier_name[i]+".tsv",sep="\t", index=False)        