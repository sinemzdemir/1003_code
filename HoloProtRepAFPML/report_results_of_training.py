import pandas as pd
import os
def report_results_of_training(representation_name,result_list,mean_result_list,classifier_name,file_name,index,classifier_type):
    path=os.path.dirname(os.getcwd())+'/results'
    if 'test'  not in os.listdir(path):
        os.makedirs(path+classifier_type,exist_ok=True)       
    #inex-1=len(class_len)
    result_len=27 
    if index==1:
        result_df = pd.DataFrame(columns=["representation_name","classifier_name", "acc", "std_acc",  "f1_mi", "std_f1_mi", "f1_ma", "std_f1_ma", "f1_we","std_f1_we","pr_mi", "std_pr_mi","pr_ma", "std_pr_ma", "pr_we", "std_pr_we", "rc_mi", "std_rc_mi","rc_ma", "std_rc_ma", "rc_we", "std_rc_we",  "hamm","std_hamm","auc","std_auc","mcc"])           
        result_df.loc[index-1] = result_list[0]
        result_df.reset_index(drop=True, inplace=True)                
        mean_result_dataframe = pd.DataFrame(columns=["representation_name","classifier_name", "acc", "std_acc",  "f1_mi", "std_f1_mi", "f1_ma", "std_f1_ma", "f1_we","std_f1_we", "pr_mi", "std_pr_mi","pr_ma", "std_pr_ma", "pr_we", "std_pr_we", "rc_mi", "std_rc_mi","rc_ma", "std_rc_ma", "rc_we", "std_rc_we","hamm", "std_hamm","auc","std_auc","mcc"])
           
        mean_result_dataframe.loc[index-1] = mean_result_list[0]
        mean_result_dataframe.reset_index(drop=True, inplace=True)
        result_df.to_csv(path+'/'+classifier_type+'/'+representation_name+'_'+file_name+'_'+classifier_type+".tsv", sep="\t", index=False)
        mean_result_dataframe.to_csv(path+'/'+classifier_type+'/'+representation_name+'_'+file_name+'_'+classifier_type+'_means'+".tsv", sep="\t", index=False)   
    
    #rep_name="_".join(representation_name)
                           
    elif (index>1):
       
        
        if(index>2):
            result_df=pd.read_csv(path+'/'+classifier_type+'/'+representation_name+'_'+file_name+classifier_type+".tsv", sep="\t")
            result_df.loc[index-1] = result_list[0]
            result_df.reset_index(drop=True, inplace=True)
            result_df.to_csv(path+'/'+classifier_type+'/'+representation_name+'_'+file_name+classifier_type+".tsv", sep="\t", index=False)
           
            mean_result_dataframe=pd.read_csv(path+'/'+classifier_type+'/'+representation_name+file_name+'_'+classifier_type+'_means'+".tsv", sep="\t")
            mean_result_dataframe.loc[index-1] = mean_result_list
            mean_result_dataframe.reset_index(drop=True, inplace=True)
            mean_result_dataframe.to_csv(path+'/'+classifier_type+'/'+representation_name +  file_name+'_'+classifier_type+'_means' + '.tsv',sep="\t", index=False)
        else:
            result_df=pd.read_csv(path+'/'+classifier_type+'/'+representation_name+'_'+file_name+classifier_type+".tsv", sep="\t")
            result_df.loc[index-1] = result_list[0]
            result_df.reset_index(drop=True, inplace=True)

            result_df.to_csv(path+'/'+classifier_type+'/'+representation_name+'_'+file_name+classifier_type+".tsv", sep="\t", index=False)
            mean_result_dataframe=pd.read_csv(path+'/'+classifier_type+'/'+representation_name+'_'+file_name+'_'+classifier_type+'_means'+".tsv", sep="\t")
            mean_result_dataframe.loc[index-1] = mean_result_list[0]
            mean_result_dataframe.reset_index(drop=True, inplace=True)
 
            mean_result_dataframe.to_csv(path+'/'+classifier_type+'/'+representation_name + file_name+'_'+classifier_type+'_means' + '.tsv',index=False) 
    pass    
 
