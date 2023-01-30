from audio2audio_models import Audio2AudioModels
from evaluation_utils.evaluate_model import evaluate_model 
import json
import pandas as pd
import sys

def evaluate_model_with_dataset(
    dataset_hparams_json: str, 
    model_separation_function,
    results_path: str):
    

    with open(dataset_hparams_json, "w") as f:
        data_hparams = json.load(f)

    dataset_df = pd.read_csv(data_hparams["csv"])
    results_df = evaluate_model(
            dataset_df=dataset_df,
            num_speakers=data_hparams["numSpeakers"],
            model_separation_function=model_separation_function,
            samplerate=data_hparams["numSpeakers"])
    
    results_df.to_csv(results_path)


