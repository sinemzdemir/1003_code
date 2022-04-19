# Example of binary classification configuration file
---
parameters:
    # choice of task, task names can be fuse_representations,prepare_datasets,model_training,model_test
    choice_of_task_name:  [model_training,model_test]
    # representation vectors concantenation
    
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
