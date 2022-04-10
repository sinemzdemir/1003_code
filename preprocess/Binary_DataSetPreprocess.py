import os
import pickle
import datetime
import tqdm
import pandas as pd

def integrate_go_lables_and_representations_for_binary(label_dataframe, dataset_name,n):
    """
    This function takes two dataframes and a dataframe name as input. First dataframe has two coloumns 'Label' and  'Entry'. 
'Label' column includes GO Terms and 'Entry' column includes UniProt IDs. Second dataframe has a column named as
'Entry' in the first column, but number of the the following columns are varying based on representation vector length.
The function integrates these dateframes based on 'Entry' column

    Parameters
    ----------
    label_dataframe: Pandas Dataframe
            Includes GO Terms and UniProt IDs of annotated proteins.
    representation_dataframe: Pandas Dataframe
            UniProt IDs of annotated proteins and representation vectors in multicolumn format.
    dataset_name: String
            The name of the output dataframe which is used for saving the dataframe to the results directory.

    Returns
    -------
    integrated_dataframe : Pandas Dataframe
            Integrated label dataframe. The dataframe has multiple number of  columns 'Label','Entry' and 'Vector'. 
            'Label' column includes GO Terms and 'Entry' column includes UniProt ID and the following columns includes features of the protein represention vector.
"""
    integrated_dataframe = pd.DataFrame(columns=['Entry', 'Vector'])
    global aspect_lst
    integrated_dataframe_list = []
    vals = dataset_name.iloc[:, 1:(len(dataset_name.columns))]
    representation_dataset = pd.DataFrame(columns=['Entry', 'Vector'])
    for index, row in tqdm.tqdm(vals.iterrows(), total=len(vals)):
        list_of_floats = [float(item) for item in list(row)]
        representation_dataset.loc[index] = [dataset_name.iloc[index]['Entry']] + [list_of_floats]
    if n==3:
        for index in range(len(aspect_lst)):
            integrated_dataframe = label_dataframe[index].merge(representation_dataset, how='inner', on='Entry')
            integrated_dataframe.drop(integrated_dataframe.filter(regex="Unname"),axis=1, inplace=True)
            integrated_dataframe_list.append(integrated_dataframe)
    elif n==1:
        integrated_dataframe = label_dataframe.merge(representation_dataset, how='inner', on='Entry')
        integrated_dataframe.drop(integrated_dataframe.filter(regex="Unname"),axis=1, inplace=True)          
        integrated_dataframe_list.append(integrated_dataframe)
    
    return integrated_dataframe_list
