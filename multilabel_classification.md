
# HoloprotRep: Protein Representation Multilabel Classification

- HoloprotRep aims to construct models for protein function prediction Holoprotrep can concate protein representations in to prepare datasets  for training and testing models
We construct a model consisting of  4 steps that can be used independently or contiguously:
We compare it with  other methods from the literature.
 1. Fuse_representations:
  - Concatenate of protein representation vectors
 2. Prepare_datasets:
  - Concantation of positive_sample_dataset and negative_sample_dataset for preperation of dataset which have "Entry" and multi columns representation vector and  save pickle format of dataset   

 3. Model_training:
  - Training of prepared data.Using models are Fully Connected Neural Network,RandomForestClassifier,SVC,KNeighborsClassifier
 5. Model_test: Trained model parameters use for make new prediction and evaluation

# Example of binary classification configuration file
---
parameters:
    ### choice of task, task names can be fuse_representations,prepare_datasets,model_training,model_test
    choice_of_task_name:  [model_training,model_test]
    ### representation vectors concantenation
    
    fuse_representations:
        representation_files: [../data/SeqVec_dataframe_multi_col.csv, ../data/tcga_embedding_dataframe_multi_col.csv]
        representation_names:  [seqvec, tcga]    
        
    prepare_datasets:  
        annotation_files:  [../data/BP_Low_Shallow.tsv]
        prepared_representation_file:  ../data/seqvec_tcga_fused_representations_dataframe_multi_col.csv
        representation_names:  [seqvec, tcga] 
    
    model_training:
        representation_names:  [seqvec, tcga]   
        auto:  True
        prepared_directory_path:  [../data/yayin_calismasi/results/]
        classifier_name:  ["Neural_Network","RandomForestClassifier"] 
    model_test:
        representation_names:  [seqvec, tcga]
        prepared_directory_path:  []              
        best_parameter_file:  [] 
    
    prediction:
       
        prepared_path:  ["../data/rep_dif_ae.csv"]
        classifier_name:  ["Fully Connected Neural Network"]         
        model_directory:  ["../results/test/modal_rep_ae_binary_classifier_Fully Connected Neural Network.pt"] 
        

---


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
