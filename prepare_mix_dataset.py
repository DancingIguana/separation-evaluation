import sys
import os
import json
import pandas as pd

from data_utils.preprocessing import generate_speech_mix_dataset

def prepare_mix_dataset(hparams_file: str):

    """
    Given the the hyperparameters of a file, run the
    generate_speech_mix_dataset function.
    """
    with open(hparams_file,"r") as f: hparams = json.load(f)

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

def prepare_multiple_mix_datasets(hparams_file_list: list):
    for hparams_file in hparams_file_list:
        prepare_mix_dataset(hparams_file)


if __name__ == "__main__":
    if len(sys.argv) != 3 or not os.path.exists(sys.argv[-1]):
        print("Specify after the command line the path to the new dataset hyperparameters.")
        exit()

    mode = sys.argv[-2]
    hparams_file_or_list = sys.argv[-1]
    if mode == "single":
        prepare_mix_dataset(hparams_file=hparams_file_or_list)
    elif mode == "multiple":
        prepare_multiple_mix_datasets(hparams_file_or_list
        )