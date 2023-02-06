"""
Script for generating the model evaluation pipelines, which 
indicate model type, model name, where to store the results
and which datasets to generate (from their hparams files) for
said evaluation.
"""
import copy
import json
import os

hparams_template = {
    "models": {
        "enhancers": [],
        "2speakers": [],
        "3speakers": []
    },
    "datasets": [],
    "deleteDatasetsAfterUse": True,
    "replaceExistingResults": False,
}

models = {
    "enhancers": ["metricgan","sepformer","mtl_mimic"],
    "2speakers": ["sepformer","resepformer"],
    "3speakers": ["sepformer"]
}

# Load lists of dataset files
with open("./hparams/mix_datasets/all_16000.json", "r") as f:
    hparams_files_16000_datasets = json.load(f)

with open("./hparams/mix_datasets/for2speakers.json","r") as f:
    hparams_files_2speakers_datasets = json.load(f)

with open("./hparams/mix_datasets/for3speakers.json","r") as f:
    hparams_files_3speakers_datasets = json.load(f)


# Enhancers evaluation pipeline
hparams_enhancers = copy.deepcopy(hparams_template)
hparams_enhancers["models"]["enhancers"] = models["enhancers"]
hparams_enhancers["datasets"] = hparams_files_16000_datasets

# 2speakers models evaluation pipeline
hparams_2speakers = copy.deepcopy(hparams_template)
hparams_2speakers["models"]["2speakers"] = models["2speakers"]
hparams_2speakers["datasets"] = hparams_files_2speakers_datasets

# 3speakers models evaluation pipeline
hparams_3speakers = copy.deepcopy(hparams_template)
hparams_3speakers["models"]["3speakers"] = models["3speakers"]
hparams_3speakers["datasets"] = hparams_files_3speakers_datasets

if not os.path.exists("./hparams/"): os.mkdir("./hparams/")
if not os.path.exists("./hparams/evaluation/"): os.mkdir("./hparams/evaluation/")

with open("./hparams/evaluation/enhancers.json", "w") as f: 
    json.dump(hparams_enhancers,f,indent=6)

with open("./hparams/evaluation/2speakers.json", "w") as f: 
    json.dump(hparams_2speakers,f,indent=6)

with open("./hparams/evaluation/3speakers.json", "w") as f: 
    json.dump(hparams_3speakers,f,indent=6)