#TODO: Script for setting up the project and moving the corresponding files into setup_utils
import sys
import json
from setup_utils import setup_minilibrispeech, setup_models, setup_evaluation_hparams, setup_mix_hparams

def setup_project(setup_type: str, project_variables_file: str) -> None:
    if setup_type == "data":
        print("Setting up MiniLibriSpeech...")
        setup_minilibrispeech(project_variables_file)

    elif setup_type == "models":
        print("Setting up models...")
        setup_models(project_variables_file)

    elif setup_type == "hparams":
        print("Setting up mixes hyperparameters...")
        setup_mix_hparams(project_variables_file)
        print("Setting up evaluation hyperparameters...")
        setup_evaluation_hparams(project_variables_file)

    elif setup_type == "eval_hparams":
        print("Setting up evaluation hyperparameters...")
        setup_evaluation_hparams(project_variables_file)

    elif setup_type == "mix_hparams":
        print("Setting up mixes hyperparameters...")
        setup_mix_hparams(project_variables_file)

    elif setup_type == "all":
        print("Setting up models...")
        setup_models(project_variables_file)
        print("\n\nSetting up MiniLibriSpeech...")
        setup_minilibrispeech(project_variables_file)
        print("\n\nSetting up data mixtures hyperparameters...")
        setup_minilibrispeech(project_variables_file)
        print("\n\nSetting up evaluation hyperparameters...")
        setup_evaluation_hparams(project_variables_file)
    print("Done!")
        

if "__name__" == "__main__":
    setup_types = ["data","hparams","eval_hparams", "mix_hparams", "models","all"]
    variables = sys.argv
    if len(variables) != 2 or variables[1] not in setup_types:
        print("Invalid arguments, structure must be as follows:")
        print("python3 setup_project.py {setup_type}")
        print("Where setup_type might be:")
        print("- data\n- hparams\n- eval_hparams\n- mix_hparams\n- all")

    setup_type = variables[1]
    project_variables_file= "./project_variables.json"
    setup_project(setup_type, project_variables_file)
