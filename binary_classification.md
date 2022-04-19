
# Example of binary classification configuration file 

parameters:

    ##choice of task, task names can be fuse_representations,prepare_datasets,model_training,model_test
    
    choice_of_task_name:  [model_training]
    
    ##representation vectors concantenation
    
    representation_names: [modal_rep_ae]
    
    fuse_representations:
    
        representation_files: [../data/HoloProtReti_modal_rep_ae_multi_col_256.csv]
        
    prepare_datasets:  
    
        positive_sample_data:  [../data/positive.csv]
        negative_sample_data:  [../data/sinem/neg_data.csv]
        prepared_representation_file:  [../data/multi_modal_rep_ae_multi_col_256.csv] 
        
    
    model_training:
    
        auto:  True
        min_fold_number:  None
        prepared_path:  [../data/results/modal_rep_ae_binary_data.pickle]
        classifier_name:  ["Fully Connected Neural Network"] 
        
    model_test:
       
        prepared_path:  []              
        best_parameter_file:  []
        
    prediction:
       
        prepared_path:  ["../data/rep_dif_ae.csv"]
        classifier_name:  ['Fully Connected Neural Network']         
        model_directory:  ["../results/test/modal_rep_ae_binary_classifier_Fully Connected Neural Network.pt"] 
        
