# HoloprotRep: Protein Representation Classification

- HoloprotRep aims to construct models for protein function prediction Holoprotrep can concate protein representations in to prepare datasets  for training and testing models
We construct a model consisting of  4 steps that can be used independently or contiguously:
We compare it with  other methods from the literature.
 1. fuse_representations:concatenate of protein representation vectors
 2. prepare_datasets:Concantation of positive_sample_dataset and negative_sample_dataset for preperation of dataset which have "Entry" and multi columns representation vector and  save pickle format of dataset   
prepare_datasets
 3. model_training:training of prepared data.Using models are Fully Connected Neural Network,RandomForestClassifier,SVC,KNeighborsClassifier
 4. model_test:
 5. model_prediction

# How to run HoloprotRep 

Step by step operation:
  1. Clone repostory
  2. Install dependencies(given below)
  3. Download datasets,unzip and place the folder
  4. Edit the configuration file (according to classification methods, for binary classification: **holoprotRep_binary_label_config.yaml** for multi label classification: **holoprotRep_multi_label_config.yaml**) by changing parameters as desired and setting paths of your file(s)
  5. cd intoholoprotrep run **HoloProtRep_binary_label.py** or **HoloProtRep_multi_label.py**
- i.e., python **HoloProtRep_binary_label.py**

# Dependencies
 1.	Python 3.7.3
 2.	pandas 1.1.4
 3.	scikit-learn 0.22.1.
 4.	Scikit-MultiLearn

- Example of binary classification configuration file see documentation [binary_classification.md](binary_classification.md), for example of multilabel classification see documentation [multilabel_classification.md](multilabel_classification.md)


