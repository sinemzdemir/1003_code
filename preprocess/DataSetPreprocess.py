"""python HoloProtRep.py -f -rf /media/DATA/home/muammer/holoprotrep/
This module produces integrated label and representation vector combinations. The module structure is the following:
- The  module implements a common ``integrate_go_lables_and_representations_for_multilabel`` method. The method takes two dataframes as input.
First dataframe has two coloumns 'Label' and  'Entry'. 'Label' column includes GO Terms and 'Entry' column includes UniProt IDs. 
Second dataframe has a coloumn named as 'Entry' as the first column, but number of the the following columns are varying based on representation vector length.
The fused representations are both saved and returned by the function.

- The  module implements a common ``convert_obo_to_dataframe`` method. The method reads a Gene Ontology (GO) ontology file
(in OBO Format) and produces three data frames for MF,BP and CC aspects.
"""
import os
import pickle
import datetime
import tqdm
import pandas as pd
from goatools.obo_parser import GODag
from goatools.rpt.rpt_lev_depth import RptLevDepth
from goatools.associations import dnld_assc
from goatools.semantic import TermCounts, get_info_content
from goatools.semantic import lin_sim
from goatools.semantic import semantic_distance
from goatools.obo_parser import GODag
aspect_lst = ["cellular_component", "biological_process", "molecular_function"]


'''def convert_obo_to_dataframe(obo_file_path, annotation_file):
    """
    This function reads a Gene Ontology (GO) ontology file (in OBO Format)and Gene Ontology annotation file (in tsv format) then  produces three 	    data frames for MF,BP and CC aspects.
    The dataframes has two columns 'Label' and 'Entry'. 'Label' column includes GO Terms and 'Entry' column includes UniProt ID
    of the annotated protein with the GO Term. The saved dataframe includes date of the OBO file such as; 'aspect_go_dataframe_DD_MM_YYYY.csv'

    Parameters
    ----------
    obo_file_path: String
            Path of the GO OBO file.
    annotation_file: Dataframe
            GO annotation file 
    Returns
    -------
    Value: DataFrame list
            This functions saves the GO dataframe, returns DataFrame list
    """

    GO_annotation_file = pd.read_csv(
        annotation_file, sep="\t", engine='python')
    len(GO_annotation_file)
    duplicateRowsDF = GO_annotation_file[GO_annotation_file.duplicated()]

    print("Duplicate Rows except first occurrence based on all columns are :")
    print(duplicateRowsDF)

    GO_annotation_file = GO_annotation_file.query('GO_EVIDENCE != "IEA"')

    GO_annotation_file.reset_index(drop=True, inplace=True)

    GO_annotation_file_human = GO_annotation_file[GO_annotation_file['TAX_ID'] == 9606]

    GO_annotation_file_human.columns

    godag = GODag(str(obo_file_path))

    go_id_aspect_dataframe = pd.DataFrame(columns=['GO_ID', 'Aspect'])

    for go_id in tqdm.tqdm(list(set(GO_annotation_file_human['GO_ID']))):
        aspect = godag[go_id].namespace
        go_id_aspect_dataframe = go_id_aspect_dataframe.append(
            {'GO_ID': go_id, 'Aspect': aspect}, ignore_index=True)

    dataframe_lst = []
    protein_go_id_dataframe = pd.DataFrame()
    global aspect_lst

    for aspect in aspect_lst:
        protein_go_id_dataframe = go_id_aspect_dataframe[go_id_aspect_dataframe['Aspect'] == aspect]
        protein_go_id_dataframe = GO_annotation_file_human[GO_annotation_file_human['GO_ID'].isin(
            protein_go_id_dataframe['GO_ID'])]
        aspect_protein_GO_id_dataframe = protein_go_id_dataframe[[
            'GO_ID', 'DB_OBJECT_ID']]
        aspect_protein_GO_id_dataframe.reset_index(drop=True, inplace=True)
        aspect_protein_GO_id_dataframe

        protein_name_group_by = aspect_protein_GO_id_dataframe.groupby(
            ['DB_OBJECT_ID'])  # protein ismi unique oldu
        selected_aspect_go_id_uniq = protein_name_group_by['GO_ID'].unique()
        selected_aspect_annotation_dataframe = pd.DataFrame(
            {'Entry': selected_aspect_go_id_uniq.index, 'Label': selected_aspect_go_id_uniq.values})
        file_save_date = datetime.datetime.now()

        #selected_aspect_annotation_dataframe.to_csv(paths +aspect+"_ "+"go_dataframe_"+datetime.datetime.strftime(file_save_date, '%d_ %B_ %Y'))
        dataframe_lst.append(selected_aspect_annotation_dataframe)
    return dataframe_lst'''


def integrate_go_lables_and_representations_for_multilabel(label_dataframe, dataset_name,n):
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
        
        concate_columns=list(set(integrated_dataframe.columns)-set({'Entry','Vector'}))
        integrated_dataframe['Label']=[j for i in integrated_dataframe.apply(lambda row: row[row == 1].index.tolist() , axis=1) for j in i]
        integrated_dataframe.drop(concate_columns,axis=1)            
        integrated_dataframe_list.append(integrated_dataframe)
    
    return integrated_dataframe_list

