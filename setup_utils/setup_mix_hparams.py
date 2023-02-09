import json
import os

def setup_mix_hparams(project_variables_file: str) -> None:
    with open(project_variables_file, "r") as f:
        project_variables = json.load(f)

    template = project_variables["templates"]["mixHparams"]
    hparams_root = project_variables["paths"]["hparamsRoot"]
    mix_dataset_hparams_root = project_variables["paths"]["mixDatasetHparamsRoot"]
    mix_datasets_root = project_variables["paths"]["mixDatasetsRoot"]
    num_speakers = project_variables["mixHparamsGenerator"]["numSpeakers"]
    mix_snr = project_variables["mixHparamsGenerator"]["mixSNR"]
    noise_snr = project_variables["mixHparamsGenerator"]["whiteNoiseSNR"]


    if not os.path.exists(hparams_root): 
        os.mkdir(hparams_root)
    if not os.path.exists(mix_dataset_hparams_root): 
        os.mkdir(mix_dataset_hparams_root)

    for num_speaker_val in num_speakers:
        for noise_snr_val in noise_snr:
            # When there's 1 speaker, we only need
            # to have variation in noise_snr_val, not mix
            if num_speaker_val == 1: 
                if not os.path.exists(os.path.join(mix_dataset_hparams_root,"1")): 
                    os.mkdir(os.path.join(mix_dataset_hparams_root,"1"))
                #We aren't interested in clean sources:
                if noise_snr_val == None: continue
                template["path"] = os.path.join(
                    mix_dataset_hparams_root,
                    f"{num_speaker_val}_{noise_snr_val}_N_16000"
                )
                template["newSampleRate"] = 16000
                template["numSpeakers"] = num_speaker_val
                template["mixAugmentationPipeline"] = [
                    {
                        "method": "addNoise",
                        "csvFile":None,
                        "snrLow":noise_snr_val,
                        "snrHigh":noise_snr_val
                    }
                ]
                with open(
                    os.path.join(
                        mix_dataset_hparams_root,
                        "1",
                        f"{noise_snr_val}_N_{template['newSampleRate']}.json"
                    ), "w") as f:
                    json.dump(template,f,indent=6)
                continue
            
            # For 2 and 3 speakers
            for mix_snr_val in mix_snr:
                temp = noise_snr_val
                if not os.path.exists(
                    os.path.join(mix_dataset_hparams_root,str(num_speaker_val))
                ):
                    os.mkdir(
                        os.path.join(mix_dataset_hparams_root,str(num_speaker_val))
                    )
                    
            
                template["numSpeakers"] = num_speaker_val
                template["mixSNR"] = mix_snr_val
                template["sourceAugmentationPipeline"] = []
                for samplerate in [8000,16000]: # When using enhancers, the samplerate will be 16000
                    template["path"] = os.path.join(
                        mix_datasets_root,
                        f"{num_speaker_val}_{noise_snr_val}_{mix_snr_val}_{samplerate}")
                    template["newSampleRate"] = samplerate


                    if noise_snr_val != None:
                        template["mixAugmentationPipeline"] = [
                            {
                                "method": "addNoise",
                                "csvFile": None, # None = white noise
                                "snrLow": noise_snr_val,
                                "snrHigh": noise_snr_val,
                            }
                        ]

                    else:
                        template["mixAugmentationPipeline"] = [] 
                        temp= "N"
                    
                    with open(
                        os.path.join(
                            mix_dataset_hparams_root,
                            str(num_speaker_val),
                            f"{temp}_{mix_snr_val}_{samplerate}.json"
                        ), "w") as f: 
                        json.dump(template, f,indent=6)
