import yaml

# import argparse
import pandas as pd
import tqdm
from preprocess import RepresentationFusion

# from preprocess import BinaryRepresentationFusion
from preprocess import DataSetPreprocess
from HoloProtRepAFPML import TrainModelsWithHyperParameterOptimization

# from HoloProtRepAFPML import AutomatedFunctionPredictionML
import os
import pickle
from HoloProtRepAFPML import Test_score_calculator
import imghdr

# upload yaml file
import glob

yaml_file_path = os.getcwd()
stream = open(yaml_file_path + "/holoprotRep_multi_label_config.yaml", "r")
data = yaml.safe_load(stream)
path = os.path.dirname(os.getcwd())
# check if results file exist
if "results" not in os.listdir(path):
    os.makedirs(path + "/results", exist_ok=True)

representation_list_of_go_ids_and_annotations_dataframe = []
best_param = []
datapreprocessed_lst = []
representation_names_list = data["parameters"]["representation_names"]
if len(representation_names_list) > 1:
    representation_names = "_".join(
        [str(representation) for representation in representation_names_list]
    )
else:
    representation_names = representation_names_list[0]

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
        """if param_fuse=="min_fold_number": 
            if(data["parameters"]["fuse_representations"]["min_fold_number"]!="None" ):
                if data["parameters"]["fuse_representations"]["min_fold_number"]>0 :
                    min_fold_num =  data["parameters"]["fuse_representations"]["min_fold_number"]
            else:
                min_fold_num =  len(representation_list_of_go_ids_and_annotations_dataframe)"""

        representation_dataframe = RepresentationFusion.produce_fused_representations(
            representation_list_of_go_ids_and_annotations_dataframe,
            min_fold_num,
            representation_names_list,
        )

        representation_dataframe.to_csv(
            path
            + "/results/"
            + representation_names
            + "_fused_representations_dataframe_multi_col.csv",
            index=False,
        )
if "prepare_datasets" in choice_of_task_name:
    if "fuse_representations" in choice_of_task_name:
        for i in range(len(data["parameters"]["prepare_datasets"]["annotation_files"])):
            list_of_go_ids_and_annotations_dataframe = pd.read_csv(
                data["parameters"]["prepare_datasets"]["annotation_files"][i], sep="\t"
            )
            name_of_go_id_with_protein_name_lst = (
                data["parameters"]["prepare_datasets"]["annotation_files"][i]
                .split(".")[0]
                .split("/")[-1]
            )
            # name_of_go_id_with_protein_name_df='_'.join(name_of_go_id_with_protein_name_lst)

            datapreprocessed_lst.append(
                DataSetPreprocess.integrate_go_lables_and_representations_for_multilabel(
                    list_of_go_ids_and_annotations_dataframe,
                    representation_dataframe,
                    1,
                )
            )
            # path=os.path.dirname(os.getcwd())

            with open(
                path
                + "/"
                + "results"
                + "/"
                + representation_names
                + "_"
                + name_of_go_id_with_protein_name_lst
                + ".pickle",
                "wb",
            ) as handle:
                pickle.dump(datapreprocessed_lst[i], handle)

    else:
        prepared_representation_file_path = data["parameters"]["prepare_datasets"][
            "prepared_representation_file"
        ]
        representation_dataframe = pd.read_csv(prepared_representation_file_path)
        path = os.path.dirname(os.getcwd())

        if len(data["parameters"]["prepare_datasets"]["annotation_files"]) == 1:
            list_of_go_ids_and_annotations_dataframe = pd.read_csv(
                data["parameters"]["prepare_datasets"]["annotation_files"][0], sep="\t"
            )
            name_of_go_id_with_protein_name_lst = (
                data["parameters"]["prepare_datasets"]["annotation_files"][0]
                .split(".")[0]
                .split("/")[-1]
            )
            list_of_go_ids_and_annotations_dataframe = (
                list_of_go_ids_and_annotations_dataframe.rename(
                    columns={"Protein_Id": "Entry"}
                )
            )
            datapreprocessed_lst.append(
                DataSetPreprocess.integrate_go_lables_and_representations_for_multilabel(
                    list_of_go_ids_and_annotations_dataframe,
                    representation_dataframe,
                    1,
                )
            )
            with open(
                path
                + "/"
                + "results"
                + "/"
                + representation_names
                + "_"
                + name_of_go_id_with_protein_name_lst
                + ".pickle",
                "wb",
            ) as handle:
                pickle.dump(datapreprocessed_lst[0][0], handle)
        else:
            for i in range(
                len(data["parameters"]["prepare_datasets"]["annotation_files"])
            ):
                list_of_go_ids_and_annotations_dataframe = pd.read_csv(
                    data["parameters"]["prepare_datasets"]["annotation_files"][i],
                    sep="\t",
                )
                name_of_go_id_with_protein_name_df = (
                    data["parameters"]["prepare_datasets"]["annotation_files"][i]
                    .split(".")[0]
                    .split("/")[-1]
                )

                datapreprocessed_lst.append(
                    DataSetPreprocess.integrate_go_lables_and_representations_for_multilabel(
                        list_of_go_ids_and_annotations_dataframe,
                        representation_dataframe,
                        1,
                    )
                )
                # name_of_go_id_with_protein_names='_'.join(name_of_go_id_with_protein_name_lst)
                with open(
                    path
                    + "/"
                    + "results"
                    + "/"
                    + representation_names
                    + "_"
                    + name_of_go_id_with_protein_name_df
                    + ".pickle",
                    "wb",
                ) as handle:
                    pickle.dump(datapreprocessed_lst[i], handle)
best_param_lst = []
if "model_training" in choice_of_task_name:
    if "prepare_datasets" in choice_of_task_name:
        for data_preproceed in datapreprocessed_lst:
            best_param = TrainModelsWithHyperParameterOptimization.select_best_model_with_hyperparameter_tuning(
                representation_names,
                name_of_go_id_with_protein_name_lst,
                data_preproceed,
                data["parameters"]["model_training"]["classifier_name"],
                data["parameters"]["model_training"]["auto"],
            )
            if "model_test" in choice_of_task_name:
                Test_score_calculator.Model_test(
                    representation_names,
                    name_of_go_id_with_protein_name_lst,
                    data_preproceed,
                    best_param,
                )

    else:

        preprocesed_data_directory_path = data["parameters"]["model_training"][
            "prepared_directory_path"
        ]
        i = 0
        for data_preproceed in glob.glob(
            preprocesed_data_directory_path[0] + "*.pickle"
        ):
            i = i + 1
            data_preproceed_pickle = open(data_preproceed, "rb")
            data_preproceed_df = pickle.load(data_preproceed_pickle)
            best_param = TrainModelsWithHyperParameterOptimization.select_best_model_with_hyperparameter_tuning(
                representation_names_list,
                data_preproceed.split(".")[0].split("/")[-1],
                data_preproceed_df,
                data["parameters"]["model_training"]["classifier_name"],
                data["parameters"]["model_training"]["auto"],
            )

            if "model_test" in choice_of_task_name:
                preprocesed_data_directory_path = data["parameters"]["model_test"][
                    "prepared_directory_path"
                ]
                for data_preproceed in glob.glob(
                    preprocesed_data_directory_path[0] + "*.pickle"
                ):
                    data_preproceed_pickle = open(data_preproceed, "rb")
                    data_preproceed_df = pickle.load(data_preproceed_pickle)
                classifier_name = data["parameters"]["model_test"]["classifier_name"][
                    i - 1
                ]
                parameter = pd.read_csv(
                    data["parameters"]["model_test"]["best_parameter_file"][0]
                )
                Test_score_calculator.Model_test(
                    representation_names, data_preproceed_df, parameter
                )

if "model_test" in choice_of_task_name:
    if len(choice_of_task_name) == 1:
        preprocesed_data_directory_path = data["parameters"]["model_test"][
            "prepared_directory_path"
        ]

        for data_preproceed in glob.glob(
            preprocesed_data_directory_path[0] + "*.pickle"
        ):

            data_preproceed_pickle = open(data_preproceed, "rb")
            data_preproceed = pickle.load(data_preproceed_pickle)
        classifier_name = "_".join(data["parameters"]["model_test"]["classifier_name"])
        parameter = pd.read_csv(
            data["parameters"]["model_test"]["best_parameter_file"][0]
        )
        Test_score_calculator.Model_test(
            representation_names, data_preproceed, parameter
        )
