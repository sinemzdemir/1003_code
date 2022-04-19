
# Example of binary classification configuration file 

parameters:

    ##choice of task, task names can be fuse_representations,prepare_datasets,model_training,model_test
    
    choice_of_task_name:  [model_training]
    
    ##protein representation model name for result files
    
    representation_names: [modal_rep_ae]
    ##representation vectors concantenation
    fuse_representations:
    
        representation_files: [../data/HoloProtReti_modal_rep_ae_multi_col_256.csv]
    ## Concantation of positive_sample_dataset and negative_sample_dataset for preperation of dataset which have "Entry" and multi columns representation vector and  save pickle format of dataset   
    prepare_datasets:  
    
        positive_sample_data:  [../data/positive.csv]
        negative_sample_data:  [../data/sinem/neg_data.csv]
        prepared_representation_file:  [../data/multi_modal_rep_ae_multi_col_256.csv] 
        
    ## Training of prepared data
    model_training:
    
        auto:  True
        ## if run training alone enter prepared path
        prepared_path:  [../data/results/modal_rep_ae_binary_data.pickle]
        ## Enter classifier name which can be "Fully Connected Neural Network","RandomForestClassifier","SVC","KNeighborsClassifier"
        classifier_name:  ["Fully Connected Neural Network"] 
        
    prediction:
        ## if run training alone enter prepared path
        prepared_path:  ["../data/rep_dif_ae.csv"]
        classifier_name:  ['Fully Connected Neural Network']         
        model_directory:  ["../results/test/modal_rep_ae_binary_classifier_Fully Connected Neural Network.pt"] 
        
