# HoloprotRep: Protein Representation Classification

- HoloprotRep aim to fuse protein representations in order to prepare datasets which will be utilized for model training and test
We construct model which comprise of 4 level which can be use independent or related:
compare it other methods from literature.

1.	fuse_representations
2.	prepare_datasets
3.	model_training
4.	model_test

# How to run HoloprotRep 

- Step by step operation:
1. Clone repostory
2. Install dependencies(given below)
3. Download datasets,unzip and place the folder
4. Edit the configuration file (according to classification methods, for binary classification: **holoprotRep_binary_label_config.yaml** for multi label classification: **holoprotRep_multi_label_config.yaml**) by changing parameters as desired and setting paths of your file(s)
5. cd intoholoprotrep run **HoloProtRep_binary_label.py** or **HoloProtRep_multi_label.py**
- i.e., python **HoloProtRep_binary_label.py**


# Example of configuration file
| **Command **            | **Value** |
|--------------------|-------|
|**choice_of_task_name:**| [fuse_representations,prepare_datasets,model_training,model_test] |
|**fuse_representations:** |   
| representation_files: | [/media/DATA/home/sinem/yayin_calismasi/SeqVec_dataframe_multi_col.csv, /media/DATA/home/sinem/yayin_calismasi/tcga_embedding_dataframe_multi_col.csv] |
|       min_fold_number: |  None |
|       representation_names: | [seqvec, tcga]    | 
|**prepare_datasets:**  |
|annotation_files: |  [/media/DATA/home/sinem/low_data/cellular_component_Low_Normal.csv, /media/DATA/home/sinem/low_data/molecular_function_Low_Normal.csv] |
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
- # Dependencies
1.	Python 3.7.3
2.	pandas 1.1.4
3.	scikit-learn 0.22.1.
4.	Scikit-MultiLearn

# Definition of output files (results)
The files described below are generated as the result/output of a HoloprotRep run. They are located under the "results" folder. 
# Default output (these files are produced in the default run mode):
# 1.fuse_representations
representation_name _dataframe_multi_col.csv: This file includes UniProt protein ids and their  multicolumn representations
# 2.prepare_datasets
fused_representations_abundance.pickle: This file includes UniProt protein ids (“Entry”), 'Label' column includes GO Terms and Vector column includes fused protein representation vector
# 3.model_training
fused_representation_name_model_names.tsv: File consist of  Accuracy, f1_micro, f1_macro, f_max, f1_weighted, precision_min, precision_max, precision_weighted, recall_min, recall_max,recall_weighted, hamming distance, Matthews correlation coefficient, Accuracy_standard_deviation, f1_min_standard_deviation, f1_max_standard_deviation,  f1_weighted_standard_deviation, precision_min_standard_deviation, precision_max_standard_deviation, precision_weighted_standard_deviation, recall_min_standard_deviation, recall_max_standard_deviation, recall_weighted_standard_deviation,hamming distance_standard_deviation, Matthews correlation coefficient_standard_deviation columns\ 
classifier_name_representation_names.sav:saved model\
representation_names_models_train_means.tsv:	trained model results means\
# 4.model_test\
representation_names_model_test__predictions.tsv:\
representation_names__test_means.tsv: has same columns with training results file\
representation_names__test.tsv: has same columns with training results file\
