from data_utils.librispeech_prepare import prepare_mini_librispeech
import json
import os


# Get hyperparams
with open("./project_variables.json", "r") as f:
    project_variables = json.load(f)
mls_params = project_variables["MLS"]

prepare_mini_librispeech(
    data_folder=mls_params["directory"],
    save_json_train=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["json_train"]),
    save_json_test=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["json_test"]),
    save_json_valid=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["json_valid"]),
    save_csv_train=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["csv_train"]),
    save_csv_test=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["csv_test"]),
    save_csv_valid=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["csv_valid"]),
)