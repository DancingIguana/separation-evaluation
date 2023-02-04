import sys

sys.path.append("./speechbrain")
import os
from audio2audio_models import Audio2AudioModels
from evaluation_utils.evaluate_model import evaluate_model 
import json
import pandas as pd

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


if __name__ == "__main__":
    if len(sys.argv) != 5 or not os.path.exists(sys.argv[-3]) or sys.argv[-2] not in ["enhancers","2speakers","3speakers"]:
        print("Invalid arguments, must include: python3 model_evaluation {results_csv_path} {data_hparams_json_path} {model_type} {model}")
        print("Example: python3 model_evaluation ./my_results.csv ./data/2_0_0/general_info.json 2speakers")
        exit()

    results_path = sys.argv[-4]
    dataset_hparams_json = sys.argv[-3]
    model_type = sys.argv[-2]
    model_name = sys.argv[-1]

    evaluate_model_with_dataset(
        dataset_hparams_json=dataset_hparams_json,
        model_type=model_type,
        model_name=model_name,
        results_path=results_path
    )
