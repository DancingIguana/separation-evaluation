from data_utils.librispeech_prepare import prepare_mini_librispeech
import json
import os


# Get hyperparams
with open("./hparams/minilibrispeech_hparams.json", "r") as f:
    data_hparams = json.load(f)


# Download and prepare the base Mini LibriSpeech dataset
prepareMLS = data_hparams["prepareMLS"]

if prepareMLS:
    mls_params = data_hparams["MLS"]
    
    prepare_mini_librispeech(
        data_folder=mls_params["directory"],
        save_json_train=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["json_train"]),
        save_json_test=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["json_test"]),
        save_json_valid=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["json_valid"]),
        save_csv_train=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["csv_train"]),
        save_csv_test=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["csv_test"]),
        save_csv_valid=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["csv_valid"]),
    )


