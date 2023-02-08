import os
import json

# For all enhancers (16k samplerate for any number of speakers)
all_16000_pipelines = [
    os.path.join(dp, f) for dp, dn, filenames 
    in os.walk("./hparams/mix_datasets/") for f in filenames 
    if "16000.json" in f
]

with open("./hparams/mix_datasets/for_enhancers.json","w") as f:
    json.dump(all_16000_pipelines, f, indent=6)


# For 2speakers
all_for_2speakers_pipelines = [
    os.path.join(dp, f) 
    for dp, dn, filenames in os.walk("./hparams/mix_datasets/") 
    for f in filenames if "8000.json" in f 
    and "2" in os.path.join(dp,f).split("/")
    or "3" in os.path.join(dp,f).split("/")
]


with open("./hparams/mix_datasets/for2speakers.json","w") as f:
    json.dump(all_for_2speakers_pipelines, f, indent=6)


# For 3speakers
all_for_3speakers_pipelines = [
    os.path.join(dp, f) 
    for dp, dn, filenames in os.walk("./hparams/mix_datasets/") 
    for f in filenames if "8000.json" in f 
    and "3" in os.path.join(dp,f).split("/")
]

with open("./hparams/mix_datasets/for3speakers.json","w") as f:
    json.dump(all_for_3speakers_pipelines, f, indent=6)