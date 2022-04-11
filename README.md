# 1003 protein_kesfi_schedule (multilabel)
HoloprotRep aim to fuse protein representations in order to prepare datasets which will be utilized for model traing and test
We construct model which comprise of 4 level which can be use independent or related
1.	fuse_representations
2.	prepare_datasets
3.	model_training
4.	model_test

# How to run HoloprotRep 

Edit the configuration file **holoprotRep_binary_label_config.yaml** by changing parameters as desired and setting paths of your file/files 

# Example of configuration file
| **Command **            | **Value** |
|--------------------|-------|
|**choice_of_task_name:**| [fuse_representations,prepare_datasets,model_training,model_test] |
|**fuse_representations:** |   
| representation_files: | [/media/DATA/home/sinem/yayin_calismasi/SeqVec_dataframe_multi_col.csv, /media/DATA/home/sinem/yayin_calismasi/tcga_embedding_dataframe_multi_col.csv] |
|       min_fold_number: |  None |
|       representation_names: | [seqvec, tcga]    | 
|**prepare_datasets:**  |
|annotation_files: |  [/media/DATA/home/sinem/low_data/cellular_component_Low_Normal.csv,/media/DATA/home/sinem/low_data/molecular_function_Low_Normal.csv] |
| prepared_representation_file: | /media/DATA/home/sinem/yayin_calismasi/results/seqvec_tcga_fused_representations_dataframe_multi_col.csv |
| representation_names: | [seqvec, tcga] |
|**model_training:** |
| representation_names:|  [seqvec, tcga]   |
| auto: | True |
| prepared_directory_path: | [/media/DATA/home/sinem/yayin_calismasi/results/] |
| classifier_name: |  ["Neural_Network","RandomForestClassifier"] |
|**model_test:**|
| representation_names: | [seqvec, tcga] |
| prepared_directory_path: |  [] |             
| best_parameter_file: |  [] |
# Dependencies
1.	Python 3.7.3
2.	pandas 1.1.4
3.	scikit-learn 0.22.1.
4.	Scikit-MultiLearn

# Definition of output files (results)
The files described below are generated as the result/output of a HoloprotRep run. They are located under the "results" folder. 
# Default output (these files are produced in the default run mode):
# 1.fuse_representations
**representation_name _dataframe_multi_col.csv": This file includes UniProt protein ids and their  multicolumn representations**
# 2.prepare_datasets
**fused_representations_abundance.pickle: This file includes UniProt protein ids (“Entry”), 'Label' column includes GO Terms and Vector column includes fused protein representation vector**
# 3.model_training
# training model and parameters;
**RandomForestClassifier:** 
parameters = {'model_classifier__estimator__n_estimators': [10,50, 100,150,200], 'model_classifier__estimator__max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11],
'model_classifier__estimator__min_samples_leaf': [1, 5, 10, 20, 100]}\
 **Neural_Network:** parameters = { ‘criterion’: BCEWithLogitsLoss,
    ‘optimizer’:SGD,epoch=200}\
 **SVC:** parameters = {'model_classifier__estimator__C': [2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1, 2, 4, 8, 16, 32, 64,128, 256, 2 ** 9, 2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13, 2 ** 14,2 ** 15],'model_classifier__estimator__gamma': [2 ** -15, 2 ** -14, 2 ** -13, 2 ** -12, 2 ** -11, 2 ** -10,2 ** -9, 2 ** -8, 2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3,2 ** -2, 2 ** -1, 1, 2, 4, 8], 'model_classifier__estimator__kernel': ['linear', 'poly', 'rbf'], 'model_classifier__estimator__max_iter': [-1, 5, 10, 10, 20, 30, 40, 50, 100]}\
 **KNeighborsClassifier:** parameters = {'model_classifier__n_neighbors': k_range, 'model_classifier__weights': ["uniform", "distance"],'model_classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'model_classifier__leaf_size': list(range(1, int(len(model_label) / 5))), 'model_classifier__p': [1, 2]}\
Parameter space search with 5 fold cross validation results 
“fused_representation_name_model_names.tsv”: File consist of  Accuracy,f1_min,f1_max,    f1_weighted,precision_min,precision_max,    precision_weighted, recall_min,recall_max,    recall_weighted,hamming distance, Matthews correlation coefficient, Accuracy_standard_deviation,f1_min_standard_deviation,f1_max_standard_deviation,  f1_weighted_standard_deviation,precision_min_standard_deviation,precision_max_standard_deviation,    precision_weighted_standard_deviation, recall_min_standard_deviation,recall_max_standard_deviation,    recall_weighted_standard_deviation,hamming distance_standard_deviation, Matthews correlation coefficient_standard_deviation columns\ 
“classifier_name_representation_names.sav”:saved model\
“representation_names_models_train_means.tsv”:	trained model results means\
# 4.model_test\
“representation_names_model_test__predictions5cv.tsv”:\
“representation_names__test_means.tsv”: has same columns with training results file\
“representation_names__test.tsv”: has same columns with training results file\
