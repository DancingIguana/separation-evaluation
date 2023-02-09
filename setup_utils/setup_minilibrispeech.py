from data_utils.librispeech_prepare import prepare_mini_librispeech
import json
import os

def setup_minilibrispeech(project_variables_file: str) -> None:
    # Get hyperparams
    with open(project_variables_file, "r") as f:
        project_variables = json.load(f)
    mls_params = project_variables["MLS"]

    prepare_mini_librispeech(
        data_folder=mls_params["directory"],
        save_json_test=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["json_test"]),
        save_csv_test=os.path.join(mls_params["directory"],"LibriSpeech",mls_params["csv_test"]),
    )
