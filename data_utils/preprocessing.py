import os
import pandas as pd
import random 
from tqdm import tqdm
import json

from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataio import write_audio
from speechbrain.dataio.legacy import ExtendedCSVDataset
from speechbrain.dataio.dataloader import make_dataloader

from speechbrain.processing import speech_augmentation
from speechbrain.processing.signal_processing import (
    compute_amplitude,
    dB_to_amplitude,
    convolve1d,
    notch_filter,
    reverberate,
)

import torch
import torch.nn.functional as F



def generate_file_groupings(files_df:pd.DataFrame, num_speakers:int) -> list:
    """
    Given a dataframe with the speaker dataset file details, generate lists
    with different files. Each list contains a file from a different speaker.
    All of the files are used. In cases where there are no more speakers available
    to complete the list but some speaker still has missing files, these will be
    complimented with random previously chosen files from different speakers.

    Parameters:
    -----------------------------
    files_df: dataframe with the details of the speaker dataset. It must contain
        the following columns.
        - ID (int or string) of the file
        - duration (float) of the audio
        - wav: the audio file path
        - wav_format: the format of the audio file
        - spk_id: ID of the speaker in the audio file

    num_speakers: the number of different speakers inside a list.

    Returns
    ------------------------------
    Lists that contain a list of the audio files. Each list contains a mixture
    of different speakers and different audio files from them.
    """
    files_df["selected"] = [False for i in range(len(files_df))]

    # First generate a dataframe containing the speaker IDs and their availability status
    speakers_df = pd.DataFrame()
    speakers = list(files_df["spk_id"].unique())
    speakers_df["speaker"] = speakers
    speakers_df["available"] = [True for i in range(len(speakers))]

    # While we have unused files:
    num_unused = len(files_df[files_df["selected"] == False])

    groupings = []
    while num_unused > 0:
        #Initialize batch
        group = []

        # Source number
        num_sources = num_speakers

        # Get list of the available speakers
        available_speakers = list(speakers_df[speakers_df["available"] == True]["speaker"])
        num_available_speakers = len(available_speakers)
        
        # If we have less available speakers than the required ones
        # choose already used speakers to complete the remaining batch elements
        if num_available_speakers < num_speakers:

            # Update the number of sources to choose from for the available speakers
            num_sources = num_available_speakers

            # Choose the number of necessary speakers to complete the batch
            unavailable_speakers = list(speakers_df[speakers_df["available"] == False]["speaker"])
            chosen_unavailable_speakers = random.sample(unavailable_speakers, num_speakers-num_available_speakers)

            # Choose a random file for each speaker 
            for speaker in chosen_unavailable_speakers:
                speaker_files = list(files_df[files_df["spk_id"] == speaker]["ID"])
                chosen_file = random.choice(speaker_files)

                # Add the file to the batch
                group.append(files_df[files_df["ID"] == chosen_file].iloc[0].to_dict())
        
        # Choose the unavailable speakers
        chosen_speakers = random.sample(available_speakers, num_sources)

        # For each speaker select an unused file
        for speaker in chosen_speakers:

            # Currently available files
            available_speaker_files = list(files_df[(files_df["selected"] == False) & (files_df["spk_id"] == speaker)]["ID"])
            num_available_files = len(available_speaker_files)

            # If there's only one available file, the speaker is unavailable after this iteration
            if(num_available_files == 1):
                speakers_df.loc[speakers_df["speaker"] == speaker,"available"] = False

            # Choose one of the files
            chosen_file = random.choice(available_speaker_files)

            # Mark the file as used
            files_df.loc[files_df["ID"] == chosen_file,"selected"] = True
            
            # Add the file
            group.append((files_df[files_df["ID"] == chosen_file].iloc[0].to_dict()))


        # Update number of unused files
        num_unused = len(files_df[files_df["selected"] == False])

        # Add batch to list
        groupings.append(group)
    
    return groupings


def add_padding_to_batch(batch:list):
    """
    Given a list of tensors of varying length add a padding of 0s to the right 
    of the tensor. All of the tensors will end up having the shape of the longest 
    one in the original list.

    Parameters:
    -----------------------
    batch: the list of tensors

    Returns:
    -----------------------
    The list of padded tensors
    """
    lengths = [len(i) for i in batch]
    max_length = max(lengths)
    padded_batch = []
    for i in range(len(batch)):
        padded_batch.append(torch.cat([batch[i],torch.zeros(max_length-len(batch[i]))]))
        
    return padded_batch


def add_padding_to_batches(batches:list):
    """
    Given a set of lists, add padding to each of the lists.
    """
    padded_batches = []
    for i in range(len(batches)):
        padded_batches.append(add_padding_to_batch(batches[i]))
    
    return padded_batches


def get_tensors_from_grouping(root: str,files_df:pd.DataFrame, groupings:list, padding:bool = True) -> torch.tensor:
    tensor_batches = []
    for batch in tqdm(groupings,total = len(groupings)):
        tensor_batch = []
        for details in batch:
            audio_file = files_df[files_df["ID"] == details["ID"]].iloc[0]["wav"]
            audio_tensor = read_audio(audio_file.format(data_root = root))
            tensor_batch.append(audio_tensor)
        tensor_batches.append(tensor_batch)

    
    if not padding:
        return tensor_batches

    padded_tensor_batches = add_padding_to_batches(tensor_batches)

    # Stack the tensors for each batch
    for i in tqdm(range(len(padded_tensor_batches))):
        padded_tensor_batches[i] = torch.stack(padded_tensor_batches[i])

    return padded_tensor_batches
    

def generate_mixed_signals(batches, speaker_count, snr_high, snr_low):
    mixed_signals = []
    babbler = speech_augmentation.AddBabble(
        speaker_count = speaker_count,
        snr_high = snr_high,
        snr_low = snr_low)
    
    for batch in tqdm(batches,total=len(batches)):
        lengths = torch.ones(speaker_count+1)
        noisy = babbler(batch,lengths)
        mixed_signals.append(noisy)
    
    return mixed_signals
        
def generate_mixture_files(
    directory_name: str, 
    source_signals: list, 
    groupings,
    mixed_signals = None, 
    single_mix = False,
    sample_folder_name_template = "sample",
    mixture_folder_name = "mixtures", 
    source_folder_name = "sources",
    samplerate = 16000):
    """
    Given the source signals and the mixed signals, store the dataset locally
    and return the general structure in a dataframe.

    Parameters:
    -----------------------
    - directory_name: the path of the directory in where to store the dataset.
    - source_signals: the list of sources.
    - groupings
    """
    # Store source and mix correspondence
    source_mix_json = {}
    mix_sources_tuples = []

    # Generate directory if necessary
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    

    n_samples = len(mixed_signals)

    # Each batch has mutiple mixes from the same sources
    n_mixes_per_sample = len(mixed_signals[0])
    n_sources_per_sample = len(source_signals[0])

    # Only keep one mix (they will be very similar)
    if single_mix: 
        n_mixes_per_sample = 1

    
    mix_column = [None for _ in range(n_samples*n_mixes_per_sample)]
    source_columns = [[None for _ in range(n_samples*n_mixes_per_sample)] for _ in range(n_sources_per_sample)]

    # For each batch of the mixed signals save all of the mixes
    mix_counter_1 = 0
    mix_counter_2 = 0
    for i in tqdm(range(n_samples)):
        # Create folder for the sample
        sample_folder_name = f"{sample_folder_name_template}{i}"
        os.mkdir(directory_name + "/" + sample_folder_name)

        os.mkdir(directory_name + "/" + sample_folder_name + "/" + mixture_folder_name)
        os.mkdir(directory_name + "/" + sample_folder_name + "/" + source_folder_name)

        # Create a list of all of the speakers in the grouping
        # Each speaker has an ID, a source file and a mix file
        source_mix_json[sample_folder_name] = [{"spk_id": j["spk_id"]} for j in groupings[i]]

        # Sources
        sample_sources_paths = [None for i in range(n_sources_per_sample)]
        for j in range(n_sources_per_sample):

            source_file_name = f"source{i}_{j}.wav"
            source_file_path = directory_name + "/" + sample_folder_name + "/" + source_folder_name + "/" + source_file_name
            write_audio(source_file_path,
                        source_signals[i][j],
                        samplerate=samplerate)
            
            # Augmented source
            # Save file source-mix
            #source_mix_json[sample_folder_name][j]["sourceFile"] = source_file_path
            #source_mix_json[sample_folder_name][j]["mixFile"] = mix_file_path

            sample_sources_paths[j] = source_file_path

        # Mixes
        # For each of the mixes inside the batch
        # Generate the audio files with Speechbrain
        for j in range(n_mixes_per_sample):
            mix_file_name = f"mix{i}_{j}.wav"
            mix_file_path = directory_name + "/" + sample_folder_name + "/" + mixture_folder_name + "/" + mix_file_name
            write_audio(mix_file_path,
                        mixed_signals[i][j],
                        samplerate=samplerate)
            
            mix_column[mix_counter_1] = mix_file_path
            for k in range(n_sources_per_sample):
                source_columns[k][mix_counter_1] = sample_sources_paths[k]

            mix_counter_1 += 1
    
    #mix_column = []
    #source_columns = [[] for i in range(n_sources)]
    #counter = 0

    #print("n_source_paths",len(source_paths))
    #print(f"Expected source paths = {len(mix_paths)}*{n_sources} = {len(mix_paths)*n_sources}")
    #print("n_sources",n_sources)
    total_rows = len(mix_column)
    id_column = [i for i in range(total_rows)]
    duration_column = [1.0 for i in range(total_rows)]
    wav_format_column = ["wav" for i in range(total_rows)]
    wav_opt_column = [None for i in range(total_rows)]

    source_mix_df_dict = {
        "ID": id_column,
        "duration": duration_column,
        "mix_wav": mix_column,
        "mix_wav_format": wav_format_column,
        "mix_wav_opt": wav_opt_column
    }

    for i in range(n_sources_per_sample):
        source_mix_df_dict[f"s{i+1}_wav"] = source_columns[i]
        source_mix_df_dict[f"s{i+1}_wav_format"] = wav_format_column
        source_mix_df_dict[f"s{i+1}_wav_opt"] = wav_opt_column

    source_mix_df = pd.DataFrame(source_mix_df_dict)
    #return source_mix_json, source_mix_df
    return source_mix_df


def generate_sound_effect_csv(
    csv_path, 
    audio_files, 
    samplerate = 16000):
    """
    Given a list of audio files, generate a csv for utilising the AddNoise 
    and AddReverb functions from Speechbrain.

    Parameters:
    ------------------------
    csv_path: the path in where to store the generated csv file.
    audio_files: the list of audio files to use in the csv.
    samplerate: the sample rate of the audio files.
    """
    ids = []
    durations = []
    audio_formats = []

    for i,file in enumerate(audio_files):
        audio_tensor = read_audio(file)
        
        # Duration of the audio file
        duration = audio_tensor.shape[0]/samplerate

        # Format of the file
        audio_format = file.split(".")[-1]

        # Append to lists
        ids.append(i)
        durations.append(duration)
        audio_formats.append(audio_format)
    
    df = pd.DataFrame(
        {
        "ID": ids, 
        "duration": durations,
         "wav": audio_files,
         "wav_format": audio_formats,
         "wav_opt": [None for i in range(len(audio_formats))]
        }
    )

    df.to_csv(csv_path,index = False)


def batch_augmentation_pipeline(
    batch:torch.tensor, 
    pipeline_params:list,
    sample_rate:int = 16000
    ) -> torch.tensor:
    """
    Given a batch of tensors, perform the indicated augmentation methods 
    to the entire batch and return it

    Parameters:
    ------------------------
    - batch: the batch of signal tensors
    - pipeline_params: the list of speech augmentation methods to apply to 
    the batch.
    - sample_rate: the sample rate of the signals inside the batch (this 
    is used for adding noise in the same sample rate)
    """
    transformed_batch = torch.clone(batch)
    lengths = torch.ones(batch.shape[0])
    for method in pipeline_params:
        if method["method"] == "addNoise":
            noisifier = speech_augmentation.AddNoise(
                csv_file = method["csvFile"],
                clean_sample_rate = sample_rate,
                snr_low = method["snrLow"],
                snr_high = method["snrHigh"])
            transformed_batch = noisifier(transformed_batch,lengths)
        if method["method"] == "addReverb":
            reverb = speech_augmentation.AddReverb(
                csv_file = method["csvFile"],
                clean_sample_rate = sample_rate)
            transformed_batch = reverb(transformed_batch, lengths)
        if method["method"] == "resample":
            resampler = speech_augmentation.Resample(
                orig_freq=method["orig_freq"], 
                new_freq=method["new_freq"]
                )
            transformed_batch = resampler(batch)

    return transformed_batch


def generate_speech_mix_dataset(
    folder_name: str, 
    original_data_root: str,
    original_dataset_csv: str, 
    num_speakers: int,
    groupings_root: str,
    recycle_groupings: bool,
    snr_high: float = 0,
    snr_low: float = 0,
    source_augmentation_pipeline: list = [],
    mix_augmentation_pipeline:list =[],
    original_samplerate: int = 16000,
    new_samplerate: int = 16000,
    ) -> tuple:
    """
    Main function for generating the dataset of mixes with possible speech 
    augmentation pipelines.

    Parameters:
    -----------------------
    - folder_name: the path in where to store the dataset (creates new directory
    if necessary).
    - original_dataset_csv: the csv path where there's the original information of
    the dataset (check the create_csv function under data_utils.librispeech_prepare)
    - num_speakers: the number of speakers inside a mixture.
    - snr_high: the lower interval limit of the SNR value to use when mixing 
    (higher SNR, lower the source signal)
    - snr_low: the higher interval limit of the SNR value to use when mixing
    - source_augmentation_pipeline: the list of methods to apply to the sources before 
    mixing.
    - mix_augmentation_pipeline: the list of methods to apply to the mixes.
    - original_samplerate: the samplerate of the files in the dataset.
    - new_samplerate: the samplerate of the new dataset to use.


    Returns:
    ----------------------
    - dataset_info: a dictionary containing hyperparameters of the dataset:
        - number of speakers in a mix
        - samplerate of the dataset
        - the source augmentation pipeline
        - the mix augmentation pipeline
    - mix_sources_df: 
    """
    if num_speakers < 1:
        print("Insufficient number of speakers. ")
        return

    if not os.path.exists(groupings_root): os.mkdir(groupings_root)
    speaker_count = num_speakers - 1

    # Get the dataset files' general information
    df = pd.read_csv(original_dataset_csv)

    # Generate groupings
    groupings_path = os.path.join(groupings_root,f"{num_speakers}_groupings.json")
    if recycle_groupings and os.path.exists(groupings_path):
        print(f"Recycling groupings from {groupings_path}")
        with open(groupings_path,"r") as f:
            groupings = json.load(f)
    else:
        print("Generating groupings...")
        groupings = generate_file_groupings(
            files_df = df, 
            num_speakers = num_speakers
            )
        with open(groupings_path,"w") as f:
            json.dump(groupings,f,indent=6)
        print(f"Groupings available at: {groupings_path}")

    # Given the groupings, get the padded tensors for the files
    print("Passing info to batches...")
    padded_tensors = get_tensors_from_grouping(
        root =original_data_root,
        files_df = df, 
        groupings = groupings)

    # Resample all of the tensors if the desired sample rate is different
    if original_samplerate != new_samplerate:
        print(f"Resampling all tensors from {original_samplerate} to {new_samplerate}")
        resampled_batches = []
        resample_pipeline = [
            {
                "method": "resample", 
                "orig_freq": original_samplerate, 
                "new_freq": new_samplerate
            }
        ]
        for batch in tqdm(padded_tensors,total = len(padded_tensors)):
            resampled_batch = batch_augmentation_pipeline(
                batch, 
                resample_pipeline)
            resampled_batches.append(resampled_batch)
        padded_tensors = resampled_batches

    # apply the transformations to the signals in the batch
    print("Applying augmentation pipeline to sources...")
    
    # Make sure to only store the augmented signals for generating the mixes
    # We don't want to use it for storing the sources
    augmented_sources = padded_tensors

    if source_augmentation_pipeline != []:
        augmented_sources = []
        for batch in tqdm(padded_tensors,total = len(padded_tensors)):
            augmented_source = batch_augmentation_pipeline(
                batch,
                source_augmentation_pipeline)
            augmented_sources.append(augmented_source)
        padded_tensors = augmented_sources


    # Generate all of the mixed signals
    mixed_signals = []
    if speaker_count > 0:
        print("Mixing signals...")
        mixed_signals = generate_mixed_signals(
            batches = augmented_sources,
            speaker_count = speaker_count,
            snr_high = snr_high,
            snr_low = snr_low)
    else:
        mixed_signals = augmented_sources.copy()
    

    # Add specified transformations to all of the mixed signals
    if mix_augmentation_pipeline != []:
        print("Applying augmentation pipeline to mixes...")
        augmented_mixed_signals = []
        for batch in tqdm(mixed_signals,total=len(mixed_signals)):
            augmented_mix = batch_augmentation_pipeline(
                batch,
                mix_augmentation_pipeline)
            augmented_mixed_signals.append(augmented_mix)
        
        mixed_signals = augmented_mixed_signals


    # Generate the dataset and get the JSON with the general information
    print("Storing dataset locally...")
    mix_sources_df = generate_mixture_files(
        folder_name,
        padded_tensors, 
        groupings,
        mixed_signals, 
        samplerate = new_samplerate)

    dataset_info = {
        "csv": os.path.join(folder_name,"dataset_info.csv"),
        "numSpeakers": num_speakers,
        "samplerate": new_samplerate,
        "sourceAugmentationPipeline": source_augmentation_pipeline,
        "mixAugmentationPipeline": mix_augmentation_pipeline,
        "mixSNRLow": None if num_speakers == 1 else snr_low,
        "mixSNRHigh": None if num_speakers == 1 else snr_high
        #"mixSources": mix_sources
    }

    return dataset_info, mix_sources_df