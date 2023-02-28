# import tensorflow as tf
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)
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


def Automated_prediction_function(model, integrated_dataframe):

    label_list = list(integrated_dataframe["Label"])
    protein_representation = integrated_dataframe.drop(["Label", "Entry"], axis=1)
    proteins = list(integrated_dataframe["Entry"])
    vectors = list(protein_representation["Vector"])
    protein_and_representation_dictionary = dict(zip(proteins, vectors))
    protein_and_label_dictionary = dict(zip(proteins, label_list))
    row = protein_representation.shape[0]
    row_val = round(math.sqrt(row), 0)
    mlt = MultiLabelBinarizer()
    labels = [tuple(arr) for arr in label_list]
    model_label = mlt.fit_transform(labels)

    protein_representation_array = np.array(
        list(protein_representation["Vector"]), dtype=float
    )
    model_label_array = np.array(model_label)
    predictions_list = []
    best_parameter_df = pd.DataFrame(
        columns={"representation_name", "classifier_name", "best parameter"}
    )
    result_dict = {}
    classifier_name_lst = []
    index = 0
    model_count = 0
    file_name = "_"
    #######################classiifier name alma
    """if  estimator==      
        model = torch.load(estimator)
        model.eval()"""
    y_pred = cross_val_predict(model, protein_representation_array, model_label, cv=kf)
    path = os.path.dirname(os.getcwd())

    label_predictions.insert(0, "protein_id", protein_name)
    label_predictions["prediction_values"] = [i for i in ls]
    label_predictions.to_csv(
        path
        + "/"
        + results
        + "/"
        + representation_name
        + "_"
        + classifier_name
        + "_"
        + "__predictions5cv.tsv",
        sep="\t",
        index=False,
    )
    """if 'results' in os.listdir(path):
        os.chdir(path+'/results')
        label_predictions.to_csv(
                representation_name + '_' + classifier_name + '_' + "__predictions5cv.tsv",
                sep="\t", index=False)
    else:
        os.makedirs(path+"/results",exist_ok=True)
        os.chdir(path+'/results')
        
        label_predictions.to_csv(
                representation_name + '_' + classifier_name + '_' + "__predictions5cv.tsv",
                sep="\t", index=False)"""
