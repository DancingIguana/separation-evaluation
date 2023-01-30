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


if __name__ == "__main__":
    if len(sys.argv) != 5 or not os.path.exists(sys.argv[-3]) or sys.argv[-2] not in ["enhancers","2speakers","3speakers"]:
        print("Invalid arguments, must include: python3 model_evaluation {results_csv_path} {data_hparams_json_path} {model_type} {model}")
        print("Example: python3 model_evaluation ./my_results.csv ./data/2_0_0/general_info.json 2speakers ")
        exit()

    results_path = sys.argv[-4]
    dataset_hparams_json = sys.argv[-3]
    model_type = sys.argv[-2]
    model_name = sys.argv[-1]

    audio_model = Audio2AudioModels(model_type=model_type,model_name=model_name)

    evaluate_model_with_dataset(
        dataset_hparams_json=dataset_hparams_json,
        model_separation_function=audio_model.audio_model_function,
        results_path=results_path
    )
