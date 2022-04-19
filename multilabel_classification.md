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
