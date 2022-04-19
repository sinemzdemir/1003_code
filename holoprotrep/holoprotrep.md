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

- # Dependencies
1.	Python 3.7.3
2.	pandas 1.1.4
3.	scikit-learn 0.22.1.
4.	Scikit-MultiLearn

- # Example of configuration file:

##Representation name (used for naming output files):
representation_name: AAC
#representation_name: LEARNED-VEC
#representation_name: T5

#Benchmarks (should be one of the "similarity","family","function","affinity","all"):
# "similarity" for running protein semantic similarity inference benchmark
# "function" for running ontology-based function prediction benchmark
# "family" for running drug target protein family classification benchmark
# "affinity" for running protein-protein binding affinity estimation benchmark
# "all" for running all benchmarks
benchmark: all

#Path of the file containing representation vectors of UniProtKB/Swiss-Prot human proteins:
representation_file_human: ../data/representation_vectors/AAC_UNIPROT_HUMAN.csv
#representation_file_human: ../data/representation_vectors/LEARNED-VEC_UNIPROT_HUMAN.csv
#representation_file_human: ../data/representation_vectors/T5_UNIPROT_HUMAN.csv

#Path of the file containing representation vectors of samples in the SKEMPI dataset: 
representation_file_affinity: ../data/representation_vectors/skempi_aac_representation_multi_col.csv
#representation_file_affinity: ../data/representation_vectors/skempi_learned-vec_representation_multi_col.csv
#representation_file_affinity: ../data/representation_vectors/skempi_t5_representation_multi_col.csv

#Semantic similarity inference benchmark dataset (should be a list that includes any combination of "Sparse", "200", and "500"):
similarity_tasks: ["Sparse","200","500"]

#Ontology-based function prediction benchmark dataset in terms of GO aspect (should be one of the following: "MF", "BP", "CC", or "All_Aspects"):
function_prediction_aspect: All_Aspects

#Ontology-based function prediction benchmark dataset in terms of size-based-splits (should be one of the following: "High", "Middle", "Low", or "All_Data_Sets")
function_prediction_dataset: All_Data_Sets

#Drug target protein family classification benchmark dataset in terms of similarity-based splits (should be a list that includes any combination of "nc", "uc50", "uc30", and "mm15")
family_prediction_dataset: ["nc","uc50","uc30","mm15"]

#Detailed results (can be True or False)
detailed_output: False


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
