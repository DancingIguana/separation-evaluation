import os
import json

# For all enhancers (16k samplerate for any number of speakers)
all_16000_pipelines = [
    os.path.join(dp, f) for dp, dn, filenames 
    in os.walk("mix_pipelines") for f in filenames 
    if "16000.json" in f
]

with open("./mix_pipelines/all_16000.json","w") as f:
    json.dump(all_16000_pipelines, f, indent=6)


# For 2speakers
all_for_2speakers_pipelines = [
    os.path.join(dp, f) 
    for dp, dn, filenames in os.walk("mix_pipelines") 
    for f in filenames if "8000.json" in f and os.path.join(dp,f).split("/")[1] in ["2","3"]
]

with open("./mix_pipelines/for2speakers.json","w") as f:
    json.dump(all_16000_pipelines, f, indent=6)


# For 3speakers
all_for_3speakers_pipelines = [
    os.path.join(dp, f) 
    for dp, dn, filenames in os.walk("mix_pipelines") 
    for f in filenames if "8000.json" in f and os.path.join(dp,f).split("/")[1] == "3"
]

with open("./mix_pipelines/for3speakers.json","w") as f:
    json.dump(all_16000_pipelines, f, indent=6)