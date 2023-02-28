import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil

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
from HoloProtRepAFPML import evaluate

from sklearn.metrics import make_scorer
from sklearn import metrics
from HoloProtRepAFPML import AutomatedFunctionPredictionML
from HoloProtRepAFPML import pytorch_network
from HoloProtRepAFPML.pytorch_network import NN
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


def intersection(real_annot, pred_annot):
    count = 0
    tn = 0
    tp = 0
    for i in range(len(real_annot)):
        if real_annot[i] == pred_annot[i]:
            if real_annot[i] == 0:
                tn += 1
            else:
                tp += 1
            count += 1

    return tn, tp


def evaluate_annotation_f_max(real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total = 0
    tn = 0
    tp = 0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue

        tn, tp = intersection(real_annots[i], pred_annots[i])
        fp = len(pred_annots[i]) - tp
        fn = len(real_annots[i]) - tn
        total += 1
        recall = tp / (1.0 * (tp + fn))
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
            print(
                "At least 2 positive samples needed for each class {0} class has {1} positive samples".format(
                    column, column_sum
                )
            )
            return False
    return True


def create_valid_kfold_object_for_multilabel_splits(X, y, kf):
    check_for_at_least_two_class_sample_exits(y)
    y_df = pd.DataFrame(y)
    sample_class_occurance = dict(zip(y_df, np.zeros(len(y_df))))
    for column in y_df:
        for fold_train_index, fold_test_index in kf.split(X, y_df):
            fold_col_sum = np.sum(y_df.iloc[fold_test_index, :][column].array)
            if fold_col_sum > 0:
                sample_class_occurance[column] += 1

    for key in sample_class_occurance:
        value = sample_class_occurance[key]
        if value < 2:
            random_state = np.random.randint(1000)
            print(
                "Random state changed since at least two positive samples are needed in different train/test folds.\
                    \nHowever, only one fold exits with positive samples for class {0}".format(
                    key
                )
            )
            print("Selected random state is {0}".format(random_state))
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            create_valid_kfold_object_for_multilabel_splits(X, y_df, kf)
        else:
            return kf


def neural_network_eval(
    f_max_cv,
    kf,
    model,
    protein_representation,
    model_label_pred_lst,
    label_lst,
    index_,
    representation_name,
    classifier_name,
    protein_and_representation_dictionary,
    file_name,
    eval_type,
    protein_name,
    path,
    parameter,
):
    best_param_list = []
    representation_name_concated = "_".join(representation_name)
    paths = (
        path
        + "/"
        + eval_type
        + "/"
        + representation_name_concated
        + "_"
        + classifier_name
        + "_"
        + "multilabel_classifier"
        + ".pt"
    )
    torch.save(model.state_dict(), paths)
    representation_name_concated = "_".join(representation_name)
    best_parameter_dataframe = pd.DataFrame(parameter)
    best_parameter_dataframe.to_csv(
        path
        + "/"
        + eval_type
        + "/"
        + "Neural_network"
        + "_"
        + representation_name_concated
        + "_"
        + "multilabel_classifier"
        + "_best_parameter"
        + ".csv",
        index=False,
    )
    evaluate.evaluate(
        kf,
        protein_representation,
        model_label_pred_lst,
        label_lst,
        f_max_cv,
        classifier_name,
        representation_name_concated,
        protein_and_representation_dictionary,
        file_name,
        index_,
        eval_type,
        "multilabel",
    )

    col_names = ["Label"]
    label_predictions = pd.DataFrame(columns=col_names)
    label_predictions.append(
        pd.DataFrame({"Label": list(np.concatenate(model_label_pred_lst))})
    )
    label_predictions.insert(0, "protein_id", protein_name)
    label_predictions.to_csv(
        path
        + "/"
        + eval_type
        + "/"
        + representation_name_concated
        + "_"
        + "multilabel_classifier"
        + "_"
        + classifier_name
        + "_"
        + eval_type
        + "_predictions.tsv",
        sep="\t",
        index=False,
    )


import ast


def Model_test(representation_name, integrated_dataframe, parameteter_file):

    parameters = parameteter_file
    label_list = [label for label in integrated_dataframe["Label"]]

    # label_list = [ast.literal_eval(label) for label in integrated_dataframe['Label']]
    protein_representation = integrated_dataframe.drop(["Label", "Entry"], axis=1)
    proteins = list(integrated_dataframe["Entry"])
    vectors = list(protein_representation["Vector"])
    protein_and_representation_dictionary = dict(zip(proteins, vectors))
    row = protein_representation.shape[0]
    row_val = round(math.sqrt(row), 0)
    mlt = MultiLabelBinarizer()
    model_label = mlt.fit_transform(label_list)
    protein_representation_array = np.array(
        list(protein_representation["Vector"]), dtype=float
    )
    model_label_array = np.array(model_label)
    f_max_cv = []
    # representation_name="_".join(representation_name_list)
    path = os.path.dirname(os.getcwd()) + "/results"
    if "test" not in os.listdir(path):
        os.makedirs(path + "/test", exist_ok=True)
    classifier_name_lst = []
    index = 0
    for i in range(len(parameters)):
        classifier_name_lst.append(parameters.iloc[i]["classifier_name"])
    # classifier_len=len( classifier_name_lst )
    file_name = "_"
    file_name.join(classifier_name_lst)

    for i in range(len(parameters)):
        label_lst = []
        model_label_pred_lst = []
        if parameters.iloc[i]["classifier_name"] == "RandomForestClassifier":
            classifier_name = "RandomForestClassifier"
            classifier_name_lst.append(classifier_name)
            # params=ast.literal_eval(parameters.iloc[i]['best parameter'][0])
            params = parameters.iloc[i]["best parameter"]
            classifier = RandomForestClassifier(
                max_depth=params["model_classifier__estimator__max_depth"],
                min_samples_leaf=params[
                    "model_classifier__estimator__min_samples_leaf"
                ],
                n_estimators=params["model_classifier__estimator__n_estimators"],
            )
            classifier_multilabel = OneVsRestClassifier(classifier, n_jobs=-1)
            model_pipline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model_classifier", classifier_multilabel),
                ]
            )
            index = index + 1
        elif parameters.iloc[i]["classifier_name"] == "SVC":
            classifier_name = "SVC"
            classifier_name_lst.append(classifier_name)
            params = ast.literal_eval(parameters.iloc[i]["best parameter"][0])
            classifier = SVC(
                C=params["model_classifier__estimator__C"],
                gamma=params["model_classifier__estimator__gamma"],
                kernel=params["model_classifier__estimator__kernel"],
                max_iter=params["model_classifier__estimator__max_iter"],
            )
            classifier_multilabel = OneVsRestClassifier(classifier, n_jobs=-1)
            model_pipline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model_classifier", classifier_multilabel),
                ]
            )
            index = index + 1
        elif parameters.iloc[i]["classifier_name"] == "KNeighborsClassifier":
            classifier_name = "KNeighborsClassifier"
            classifier_name_lst.append(classifier_name)
            params = ast.literal_eval(parameters.iloc[i]["best parameter"][0])
            up_limit = int(math.sqrt(int(len(model_label) / 5)))
            k_range = list(range(1, up_limit))
            classifier = KNeighborsClassifier(
                n_neighbors=params["model_classifier__n_neighbors"],
                weights=params["model_classifier__weights"],
                algorithm=params["model_classifier__algorithm"],
                leaf_size=params["model_classifier__leaf_size"],
            )
            model_pipline = Pipeline(
                [("scaler", StandardScaler()), ("model_classifier", classifier)]
            )
            index = index + 1
        elif parameters.iloc[i]["classifier_name"] == "Fully Connected Neural Network":
            m = 0
            classifier_name = "Fully Connected Neural Network"
            classifier_name_lst.append(classifier_name)
            input_size = len(protein_representation_array[0])
            for ii in model_label:
                if len(ii) > m:
                    class_number = len(ii)
            index = index + 1

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        kf = create_valid_kfold_object_for_multilabel_splits(
            protein_representation, model_label, kf
        )

        if classifier_name == "Fully Connected Neural Network":

            input_size = len(protein_representation_array[0])
            m = 0
            protein_name = []

            (
                f_max_cv_train,
                f_max_cv_test,
                model,
                model_label_pred_lst,
                label_lst,
                protein_name_tr,
                parameter,
                protein_name,
                parameter,
                model_label_pred_test_lst,
                label_lst_test,
            ) = NN(
                kf,
                protein_representation,
                model_label,
                input_size,
                representation_name,
                protein_and_representation_dictionary,
            )

            neural_network_eval(
                f_max_cv_test,
                kf,
                model,
                protein_representation,
                model_label_pred_test_lst,
                label_lst_test,
                index,
                representation_name,
                classifier_name,
                protein_and_representation_dictionary,
                file_name,
                "test",
                protein_name,
                path,
                parameter,
            )

            paths = (
                path
                + "/test"
                + "/"
                + representation_name
                + "_"
                + classifier_name
                + ".pt"
            )
            torch.save(model, paths)

        else:
            protein_name = []
            model_label_pred = cross_val_predict(
                model_pipline,
                protein_representation_array,
                model_label,
                cv=kf,
                n_jobs=-1,
            )
            for fold_train_index, fold_test_index in kf.split(
                protein_representation, model_label
            ):

                model_label_pred_lst.append(model_label_pred[fold_test_index])
                label_lst.append(model_label[fold_test_index])
                for vec in protein_representation_array[fold_test_index]:
                    for (
                        protein,
                        vector,
                    ) in protein_and_representation_dictionary.items():
                        if str(vector) == str(list(vec)):
                            protein_name.append(protein)
                            continue

                fmax = 0.0
                tmax = 0.0
                for t in range(1, 101):
                    threshold = t / 100.0
                    fscore = evaluate_annotation_f_max(
                        model_label[fold_test_index], model_label_pred[fold_test_index]
                    )
                    if fmax < fscore:
                        fmax = fscore
                        tmax = threshold
                f_max_cv.append(fmax)

            col_names = mlt.classes_
            label_predictions = pd.DataFrame(
                np.concatenate(model_label_pred_lst), columns=col_names
            )
            ls = np.concatenate(model_label_pred_lst)
            label_predictions.insert(0, "protein_id", protein_name)
            label_predictions["prediction_values"] = [i for i in ls]
            label_predictions.to_csv(
                path
                + "/"
                + "test"
                + "/"
                + representation_name
                + "_"
                + classifier_name
                + "_test"
                + "_predictions.tsv",
                sep="\t",
                index=False,
            )
            evaluate.evaluate(
                kf,
                protein_representation,
                model_label_pred_lst,
                label_lst,
                f_max_cv,
                classifier_name,
                representation_name,
                protein_and_representation_dictionary,
                file_name,
                index,
                "test",
                "multilabel",
            )
