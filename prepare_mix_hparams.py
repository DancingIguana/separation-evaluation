import json
import os

original_data_samplerate = 16000
new_sample_rate = 8000
num_speakers = [1,2,3]
mix_snr = [-5,0,5,10,15]
noise_snr = [None,-5,0,5,10,15]

template = {
    "path": "",
    "dataRoot": "./data/MiniLibriSpeech",
    "originalDatasetCsv": "./data/MiniLibriSpeech/LibriSpeech/test.csv",
    "originalDataSamplerate": original_data_samplerate,
    "newSamplerate": new_sample_rate,
    "numSpeakers": 0,
    "mixSNR": 0,
    "sourceAugmentationPipeline": [],
    "mixAugmentationPipeline": [],
}

if not os.path.exists("./hparams/"): 
    os.mkdir("./hparams/")
if not os.path.exists("./hparams/mix_datasets/"):
    os.mkdir("./hparams/mix_datasets/")

for num_speaker_val in num_speakers:
    for noise_snr_val in noise_snr:

        # When there's 1 speaker, we only need
        # to have variation in noise_snr_val, not mix
        if num_speaker_val == 1: 
            if not os.path.exists("./hparams/mix_datasets/1"): 
                os.mkdir("./hparams/mix_datasets/1")
            #We aren't interested in clean sources:
            if noise_snr_val == None: continue
            template["path"] = f"./data/{num_speaker_val}_{noise_snr_val}_N_16000"
            template["newSamplerate"] = 16000
            template["numSpeakers"] = num_speaker_val
            template["mixAugmentationPipeline"] = [
                {
                    "method": "addNoise",
                    "csvFile":None,
                    "snrLow":noise_snr_val,
                    "snrHigh":noise_snr_val
                }
            ]
            with open(f"./hparams/mix_datasets/1/{noise_snr_val}_N_{template['newSamplerate']}.json", "w") as f:
                json.dump(template,f,indent=6)
            continue
        
        # For 2 and 3 speakers
        for mix_snr_val in mix_snr:
            temp = noise_snr_val
            if not os.path.exists(f"./hparams/mix_datasets/{num_speaker_val}"):
                os.mkdir(f"./hparams/mix_datasets/{num_speaker_val}")
                
           
            template["numSpeakers"] = num_speaker_val
            template["mixSNR"] = mix_snr_val
            template["sourceAugmentationPipeline"] = []
            for samplerate in [8000,16000]: # When using enhancers, the samplerate will be 16000
                template["path"] = f"./data/{num_speaker_val}_{noise_snr_val}_{mix_snr_val}_{samplerate}"
                template["newSamplerate"] = samplerate


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
                
                with open(f"./hparams/mix_datasets/{num_speaker_val}/{temp}_{mix_snr_val}_{samplerate}.json", "w") as f:
                    json.dump(template, f,indent=6)
