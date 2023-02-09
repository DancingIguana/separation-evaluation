#TODO: Script for setting up the project and moving the corresponding files into setup_utils
import sys
import json
from setup_utils import setup_minilibrispeech, setup_models, setup_evaluation_hparams, setup_mix_hparams

if "__init__" == "__main__":
    # Read args: data, hparams, eval_hparams, mix_hparams, all
    setup_types = ["data","hparams","eval_hparams", "mix_hparams", "models","all"]
    variables = sys.argv
    if len(variables) != 2 or variables[1] not in setup_types:
        print("Invalid arguments, structure must be as follows:")
        print("python3 setup_project.py {setup_type}")
        print("Where setup_type might be:")
        print("- data\n- hparams\n- eval_hparams\n- mix_hparams\n- all")

    
    setup_type = variables[1]
    project_variables_file= "./project_variables.json"
    #Setup data
    if setup_type == "data":
        setup_minilibrispeech(project_variables_file)
    elif setup_type == "models":
        setup_models(project_variables_file)
    elif setup_type == "hparams":
        setup_mix_hparams(project_variables_file)
        setup_evaluation_hparams(project_variables_file)
    elif setup_type == "eval_hparams":
        setup_evaluation_hparams(project_variables_file)
    elif setup_type == "mix_hparams":
        setup_mix_hparams(project_variables_file)
    elif setup_type == "all":
        print("Setting up models...")
        setup_models(project_variables_file)
        print("Setting up MiniLibriSpeech...")
        setup_minilibrispeech(project_variables_file)
        print("Setting up data mixtures hyperparameters...")
        setup_minilibrispeech(project_variables_file)
        print("Setting up evaluation hyperparameters...")
        setup_evaluation_hparams(project_variables_file)
        print("Project successfully setup")
        