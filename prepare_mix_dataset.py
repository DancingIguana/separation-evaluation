import sys
import os
import json
import pandas as pd

from data_utils.preprocessing import generate_speech_mix_dataset

print(sys.argv)
if len(sys.argv) != 2 or not os.path.exists(sys.argv[-1]):
    print("Specify after the command line the path to the new dataset hyperparameters.")
    exit()

hparams_file = sys.argv[-1]
with open(hparams_file,"r") as f:
    hparams = json.load(f)

dataset_info, dataset_df = generate_speech_mix_dataset(
    folder_name=hparams["path"],
    original_data_root = hparams["dataRoot"],
    original_dataset_csv=hparams["originalDatasetCsv"],
    num_speakers=hparams["numSpeakers"],
    snr_high=hparams["mixSNR"],
    snr_low=hparams["mixSNR"],
    source_augmentation_pipeline=hparams["sourceAugmentationPipeline"],
    mix_augmentation_pipeline=hparams["mixAugmentationPipeline"],
    original_samplerate=hparams["originalDataSamplerate"],
    new_samplerate=hparams["newSamplerate"]
)


with open(os.path.join(hparams["path"],"general_info.json"), "w") as f:
    json.dump(dataset_info,f,indent=6)

dataset_df.to_csv(os.path.join(hparams["path"],"dataset_info.csv"))

#train_info, train_df = generate_speech_mix_dataset(
#    folder_name="./data/"
#)