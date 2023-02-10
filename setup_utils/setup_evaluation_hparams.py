import copy
import json
import os

def setup_evaluation_hparams(project_variables_file: str) -> None:
    with open(project_variables_file, "r") as f: 
        project_variables = json.load(f)

    hparams_root = project_variables["paths"]["hparamsRoot"]
    evaluation_hparams_root = project_variables["paths"]["evaluationHparamsRoot"]
    mix_dataset_hparams_root = project_variables["paths"]["mixDatasetHparamsRoot"]
    hparams_template = project_variables["templates"]["evaluationHparams"]
    models = project_variables["models"]

    # Load lists of dataset files
    hparams_files_16000_datasets = [
        os.path.join(dp, f) for dp, dn, filenames 
        in os.walk(mix_dataset_hparams_root) for f in filenames 
        if "16000.json" in f
    ]

    hparams_files_2speakers_datasets = [
        os.path.join(dp, f) 
        for dp, dn, filenames in os.walk(mix_dataset_hparams_root) 
        for f in filenames if "8000.json" in f 
        and ("2" in os.path.join(dp,f).split("/")
        or "3" in os.path.join(dp,f).split("/"))
    ]

    # For 3speakers
    hparams_files_3speakers_datasets = [
        os.path.join(dp, f) 
        for dp, dn, filenames in os.walk(mix_dataset_hparams_root) 
        for f in filenames if "8000.json" in f 
        and "3" in os.path.join(dp,f).split("/")
    ]


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

    enhancers_json = os.path.join(evaluation_hparams_root,"enhancers.json")
    for2speakers_json = os.path.join(evaluation_hparams_root,"2speakers.json")
    for3speakers_json = os.path.join(evaluation_hparams_root,"3speakers.json")
    with open(enhancers_json, "w") as f: 
        json.dump(hparams_enhancers,f,indent=6)

    with open(for2speakers_json, "w") as f: 
        json.dump(hparams_2speakers,f,indent=6)

    with open(for3speakers_json, "w") as f: 
        json.dump(hparams_3speakers,f,indent=6)

    with open(os.path.join(evaluation_hparams_root,"evaluation_hparams_files.json"),"w") as f:
        json.dump([enhancers_json,for2speakers_json,for3speakers_json],f,indent=6)
