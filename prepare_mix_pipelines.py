import json

original_data_samplerate = 16000
new_sample_rate = 8000
num_speakers = [1,2,3]
snr = [-5,0,5,10,15]
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

for num_speaker_val in num_speakers:
    for snr_val in snr:
        for noise_snr_val in noise_snr:
            template["path"] = f"./data/{num_speaker_val}_{snr_val}_{noise_snr_val}"
            template["numSpeakers"] = num_speaker_val
            template["mixSNR"] = snr_val
            template["sourceAugmentationPipeline"] = []
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
                noise_snr_val= "N"
            
            with open(f"./pipelines/mix_pipelines/{num_speaker_val}_{snr_val}_{noise_snr_val}.json", "w") as f:
                json.dump(template, f,indent=6)
