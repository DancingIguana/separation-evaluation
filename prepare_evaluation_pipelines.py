"""
Script for generating the model evaluation pipelines, which 
indicate model type, model name, where to store the results
and which datasets to generate (from their hparams files) for
said evaluation.
"""
import copy
import json
import os

with open("./project_variables.json","r") as f: 
    project_variables = json.load(f)

hparams_root = project_variables["paths"]["hparamsRoot"]
evaluation_hparams_root = project_variables["paths"]["hparamsRoot"]
mix_dataset_hparams_root = project_variables["paths"]["mixDatasetHparamsRoot"]
hparams_template = project_variables["templates"]["evaluationHparams"]
models = {
    "enhancers": ["metricgan","sepformer-wham16k","mtl_mimic"],
    "2speakers": ["sepformer","resepformer"],
    "3speakers": ["sepformer"]
}

# Load lists of dataset files
with open(os.path.join(mix_dataset_hparams_root,"for_enhancers.json"), "r") as f:
    hparams_files_16000_datasets = json.load(f)

with open(os.path.join(mix_dataset_hparams_root,"for2speakers.json"),"r") as f:
    hparams_files_2speakers_datasets = json.load(f)

with open(os.path.join(mix_dataset_hparams_root,"for3speakers.json"),"r") as f:
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

if not os.path.exists(hparams_root): 
    os.mkdir(hparams_root)
if not os.path.exists(evaluation_hparams_root): 
    os.mkdir(evaluation_hparams_root)

with open(os.path.join(evaluation_hparams_root,"enhancers.json"), "w") as f: 
    json.dump(hparams_enhancers,f,indent=6)

with open(os.path.join(evaluation_hparams_root,"2speakers.json"), "w") as f: 
    json.dump(hparams_2speakers,f,indent=6)

with open(os.path.join(evaluation_hparams_root,"3speakers.json"), "w") as f: 
    json.dump(hparams_3speakers,f,indent=6)