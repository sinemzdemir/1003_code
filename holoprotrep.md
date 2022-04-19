# HoloprotRep: Protein Representation Classification

- HoloprotRep aim to fuse protein representations in order to prepare datasets which will be utilized for model training and test
We construct model which comprise of 4 level which can be use independent or related:
compare it other methods from literature.

  1.	fuse_representations
  2.	prepare_datasets
  3.	model_training
  4.	model_test

# How to run HoloprotRep 

Step by step operation:
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

- Example of binary classification configuration file see documentation [binary_classification.md](binary_classification.md), for example of multilabel classification please read multilabel_classification.md


