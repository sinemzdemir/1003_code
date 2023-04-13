import yaml
import pandas as pd
import tqdm
from preprocess import Binary_DataSetPreprocess
from preprocess import RepresentationFusion
from HoloProtRepAFPML import BinaryTrainModelsWithHyperParameterOptimization
from sklearn.utils import shuffle
import os
import pickle
from HoloProtRepAFPML import binary_prediction
from HoloProtRepAFPML import binary_Test_score_calculator
import imghdr
import ModelParameterClass as Model_parameter
# upload yaml file
yaml_file_path = os.getcwd()
stream = open(yaml_file_path + "/holoprotRep_binary_label_config.yaml", "r")
data = yaml.safe_load(stream)
path = os.path.dirname(os.getcwd())
# check if results file exist
if "results" not in os.listdir(path):
    os.makedirs(path + "/results", exist_ok=True)
parameter_class_obj=Model_parameter.ModelParameterClass(data["parameters"]["choice_of_task_name"],
    data["parameters"]["fuse_representations"],data["parameters"]["prepare_datasets"],
    data["parameters"]["model_training"],data["parameters"]["prediction"])
    
if "fuse_representations" in parameter_class_obj.choice_of_task_name:
    
    representation_dataframe=parameter_class_obj.make_fuse_representation()
datapreprocessed_lst = []
representation_names_list = parameter_class_obj.fuse_representations[
                "representation_names"
            ]
representation_names = "_".join([str(representation) for representation in representation_names_list])
'''representation_list_of_go_ids_and_annotations_dataframe = []
best_param = []


choice_of_task_name = data["parameters"]["choice_of_task_name"]

if "fuse_representations" in choice_of_task_name:
    for param_fuse in data["parameters"]["fuse_representations"].keys():
        if param_fuse == "representation_files":
            for rep_file in data["parameters"]["fuse_representations"][
                "representation_files"
            ]:
                rep_file_name = rep_file.split("/")[-1]
                print("loading " + rep_file_name + "...")
                representation_list_of_go_ids_and_annotations_dataframe.append(
                    pd.read_csv(rep_file)
                )
        if param_fuse == "min_fold_number":
            if data["parameters"]["fuse_representations"]["min_fold_number"] != "None":
                if data["parameters"]["fuse_representations"]["min_fold_number"] > 0:
                    min_fold_num = data["parameters"]["fuse_representations"][
                        "min_fold_number"
                    ]
            else:
                min_fold_num = len(
                    representation_list_of_go_ids_and_annotations_dataframe
                )
        if param_fuse == "representation_names":
            representation_names_list = data["parameters"]["fuse_representations"][
                "representation_names"
            ]
            representation_names = "_".join(
                [str(representation) for representation in representation_names_list]
            )
            representation_dataframe = (
                RepresentationFusion.produce_fused_representations(
                    representation_list_of_go_ids_and_annotations_dataframe,
                    min_fold_num,
                    representation_names_list,
                )
            )
            representation_dataframe.to_csv(
                path
                + "/results/"
                + representation_names
                + "_binary_fused_representations_dataframe_multi_col.csv",
                index=False,
            )'''


if "prepare_datasets" and "fuse_representations" in parameter_class_obj.choice_of_task_name:
   
    positive_sample_dataframe = pd.read_csv(parameter_class_obj.prepare_datasets["positive_sample_data"][0]
            
        )
    negative_sample_dataframe = pd.read_csv(parameter_class_obj.prepare_datasets["negative_sample_data"][0]
            
        )
    negative_sample_dataframe["Label"] = [0] * len(negative_sample_dataframe)
    positive_sample_dataframe["Label"] = [1] * len(positive_sample_dataframe)
    sample_dataframe = negative_sample_dataframe.append(
            positive_sample_dataframe, ignore_index=True
        )
        
    datapreprocessed_lst.append(
            Binary_DataSetPreprocess.integrate_go_lables_and_representations_for_binary(
                shuffle(sample_dataframe), representation_dataframe, 1
            )
        )
    with open(
            path
            + "/"
            + "results"
            + "/"
            + representation_names
            + "_"
            + "binary_data"
            + ".pickle",
            "wb",
        ) as handle:
            pickle.dump(datapreprocessed_lst[0], handle)

else:
        prepared_representation_file_path =parameter_class_obj.prepare_datasets["prepared_representation_file"][0]
        representation_dataframe = pd.read_csv(prepared_representation_file_path)
        path = os.path.dirname(os.getcwd())
        representation_names_list = parameter_class_obj.prepare_datasets[
            "representation_names"
        ]
        if len(parameter_class_obj.prepare_datasets["representation_names"]) > 1:
            representation_names = "_".join(representation_names_list)
        else:
            representation_names = parameter_class_obj.prepare_datasets[
                "representation_names"
            ][0]
        positive_sample_dataframe = pd.read_csv(
            parameter_class_obj.prepare_datasets["positive_sample_data"][0]
        )
        negative_sample_dataframe = pd.read_csv(
            parameter_class_obj.prepare_datasets["negative_sample_data"][0]
        )
        negative_sample_dataframe["Label"] = [0] * len(negative_sample_dataframe)
        positive_sample_dataframe["Label"] = [1] * len(positive_sample_dataframe)
        sample_dataframe = negative_sample_dataframe.append(
            positive_sample_dataframe, ignore_index=True
        )
       
        datapreprocessed_lst.append(
            Binary_DataSetPreprocess.integrate_go_lables_and_representations_for_binary(
                shuffle(sample_dataframe), representation_dataframe, 1
            )
        )
        with open(
            path
            + "/"
            + "results"
            + "/"
            + representation_names
            + "_"
            + "binary_data"
            + ".pickle",
            "wb",
        ) as handle:
            pickle.dump(datapreprocessed_lst[0], handle)

best_param_lst = []

if "model_training" in parameter_class_obj.choice_of_task_name:
    scoring_func =  parameter_class_obj.model_training["scoring_function"]
    
    if "prepare_datasets" in parameter_class_obj.choice_of_task_name:
        for data_preproceed in datapreprocessed_lst:

            best_param = BinaryTrainModelsWithHyperParameterOptimization.select_best_model_with_hyperparameter_tuning(
                representation_names,
                data_preproceed,
                scoring_func,
                parameter_class_obj.model_training["classifier_name"],
            )

            if "model_test" in choice_of_task_name:
                binary_Test_score_calculator.Model_test(
                    representation_names, data_preproceed, best_param
                )

    else:
        import glob

        preprocesed_data_path = parameter_class_obj.model_training["prepared_path"]
        representation_names_list = parameter_class_obj.prepare_datasets[
            "representation_names"
        ]
        representation_names = "_".join(representation_names_list)
        for data_preproceed in preprocesed_data_path:
            data_preproceed_pickle = open(data_preproceed, "rb")
            data_preproceed_df = pickle.load(data_preproceed_pickle)
            best_param = BinaryTrainModelsWithHyperParameterOptimization.select_best_model_with_hyperparameter_tuning(
                parameter_class_obj.model_training["representation_names"],
                data_preproceed_df,
                scoring_func,
                parameter_class_obj.model_training["classifier_name"],
            )

if "prediction" in choice_of_task_name:

    for i in parameter_class_obj.prediction["prepared_path"]:
        test_data = pd.read_csv(i)
        """rep_data = test_data.drop(["Entry","Unnamed: 0"], axis=1)
        vals = test_data.iloc[:, 2:(len(test_data.columns))]
        representation_dataset = pd.DataFrame(columns=['Entry', 'Vector'])
        for index, row in tqdm.tqdm(vals.iterrows(), total=len(vals)):
            list_of_floats = [float(item) for item in list(row)]
            representation_dataset.loc[index] = [test_data.iloc[index]['Entry']] + [list_of_floats]"""
        classifier_name_lst = parameter_class_obj.prediction["classifier_name"]
        # representation_dataset.to_csv("/media/DATA/home/sinem/yayin_calismasi/results/representation_two_col_modal_rep_ae.csv")
        binary_prediction.make_prediction(
            parameter_class_obj.prediction["representation_names"][0],
            test_data,
            parameter_class_obj.prediction["model_directory"],
            classifier_name_lst,
        )
