import sys
sys.path.append("./speechbrain")

import json
import os
import pandas as pd
import shutil

from audio2audio_models import Audio2AudioModels
from evaluation_utils.evaluate_model import evaluate_model 
from prepare_mix_dataset import prepare_mix_dataset



def evaluate_model_with_dataset(
    dataset_hparams_json: str, 
    model_name: str,
    model_type: str,
    results_path: str):
    """
    Given a file of hyperparameters of the data, the model's type and name, and
    the path where to store the results; evaluate said model with the dataset 
    specified in the hyperparameters.

    By the end of the function, the results will be displayed in a csv specifying
    the results given by the function of evaluation_utils.evaluate_model.

    Parameters:
    ------------------------
    - dataset_hparams_json: the path to the hparams of the dataset, these must 
        include:
        - csv: csv file indicating the mix file and its corresponding source 
            files
        - numSpeakers: the number of speakers in each mix
        - samplerate: the samplerate of the mixes in the dataset
        - sourceAugmentationPipeline: the pipeline of augmentation methods 
            applied to the sources in each mix
        - mixAugmentationPipeline: the pipeline of mix methods applied to the 
            overall mix.
    """
    audio_model = Audio2AudioModels(model_type=model_type,model_name=model_name)

    results_df = evaluate_model(
            dataset_hparams_json = dataset_hparams_json,
            model_separation_function=audio_model.audio_model_function,
            model_name=model_name)
    
    results_df["model_type"] = [model_type for i in range(len(results_df))]
    results_df["model_name"] = [model_name for i in range(len(results_df))]
    results_df.to_csv(results_path)
    

def multi_evaluation(multi_evaluation_json, results_root = "./results/"):

    with open(multi_evaluation_json, "r") as f: hparams = json.load(f)

    if not os.path.exists(results_root): os.mkdir(results_root)

    for hparams_file in hparams["datasets"]:
        with open(hparams_file,"r") as f: data_hparams = json.load(f)

        # If the dataset already exists, delete it or keep it if indicated
        if os.path.exists(data_hparams["path"]):
            if hparams["replaceExistingDatasets"]:
                print("Replacing existing dataset")
                shutil.rmtree(data_hparams["path"])
                prepare_mix_dataset(hparams_file)
            else:
                print("Dataset already exists, skipping generation")
        else:
            # Prepare the dataset
            prepare_mix_dataset(hparams_file)
        
        # Evaluate the same dataset in all of the specified models
        for model_type in hparams["models"]:
            for model_name in hparams["models"][model_type]:
                results_file = os.path.join(results_root,f"{model_type}_{model_name}_{data_hparams['path'].split('/')[-1]}.csv")
                if os.path.exists(results_file) and not hparams["replaceExistingResults"]:
                    print(f"{model_type} {model_name} previously evaluated with results in {results_file}. Skipping evaluation.")
                    continue
                print(f"Evaluating {model_type} {model_name} with dataset {data_hparams['path']}")
                evaluate_model_with_dataset(
                    dataset_hparams_json = os.path.join(data_hparams["path"], "general_info.json"),
                    model_name = model_name,
                    model_type = model_type,
                    results_path = results_file
                )
                print(f"Results stored in {results_file}")
        
        if hparams["deleteDatasetsAfterUse"]: shutil.rmtree(data_hparams["path"])

if __name__ == "__main__":
    if len(sys.argv) !=2:
        print("Invalid arguments, structure is as follows:")
        print("python3 model_evaluation.py {hparams_file.json}")
        print("Example: python3 model_evaluation.py ./hparams/evaluation/enhancers.json")
        exit()

    eval_hparams_json = sys.argv[1]
    multi_evaluation(multi_evaluation_json=eval_hparams_json)