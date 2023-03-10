import pandas as pd
import json
from speechbrain.dataio.dataio import read_audio
import mir_eval
import torch
import numpy as np
import itertools
from tqdm import tqdm

def evaluate_model(
    dataset_hparams_json: str,
    model_separation_function,
    model_name: str, 
    ) -> tuple:
    """
    Given a dataframe indicating the mix column and its corresponding sources,
    and a separation or enhancement model, evaluate its performance by using
    the specified data in the dataframe. This will return different variables 
    regarding the estimations from the models when compared to the original
    sources. Which are:
    - SDR
    - SIR
    - SAR
    - Time of execution in seconds
    - Required memory

    The results are stored in a dataframe and returned.

    Parameters:
    ----------------------
    - dataset_hparams_json: the path of the hyperparameters file of the 
        dataset.
    - model_separation_function: the function used to separate the noisy batch
        with Speechbrain. This function must return a tuple containing: 
            (clean_batch,time,memory,perm).
    - model_name: variable of the model name used for an edge case with 
        mtl_mimic.

    Returns:
    ----------------------
    Dataframe with the following columns:
    - mix_file: path of the mix used when running the model
    - mix_duration: duration of the mix in seconds
    - mix_samplerate: the samplerate of the mix
    - num_speakers_in_mix: the number of speakers present in the mix
    - mix_snr_low: the SNR value in the lower interval for the non-main 
        sources/speakers in the mix
    - mix_snr_low: the SNR value in the higher interval for the non-main 
        sources/speakers in the mix
    - white_noise_snr_low: the SNR value in the lower interval for the
        white noise present in the mix
    - white_noise_snr_high: the SNR vallue in the higher interval for the
        white noise present in the mix.
    - original_source: the corresponding file of the original source given
        the estimated source obtained from the model (depending on SIR/SDR).
    - main_source: 'True' if the source in the mix is not an extra in the 
        signal, meaning that the rest of the speakers vary according to the 
        mix_snr_low and mix_snr_high variables. 'False' if otherwise.
    - separation_time: the time it took for the Speechbrain model to run 
        with the corresponding mix in seconds.
    - occupied_memory: the amount of memory occupied while running the 
        Speechbrain model in bytes.
    - SIR: the estimated SIR as specified by mir_eval.
    - SDR: the estimated SDR as specified by mir_eval.
    - SAR: the estimated SAR as specified by mir_eval.
    """
    with open(dataset_hparams_json,"r") as f:
        data_hparams = json.load(f)

    num_speakers = data_hparams["numSpeakers"]
    samplerate = data_hparams["samplerate"]
    dataset_df = pd.read_csv(data_hparams["csv"])
    mix_wav = []
    original_source = []
    mix_duration = []
    start_time = []
    end_time = []
    sdr = []
    sir = []
    sar = []
    memory = []
    main_source = []
    for index, row in tqdm(dataset_df.iterrows(),total=len(dataset_df)):

        # Get the mix file and the corresponding sources
        mix_file = row["mix_wav"]
        source_files = [row[f"s{i+1}_wav"] for i in range(num_speakers)]
        
        mix = read_audio(mix_file)
        sources = [read_audio(source_file) for source_file in source_files]

        # Exclusive case because mtl_mimic changes the size of tensors according
        # to: size = (real_size//256)*256
        if model_name == "mtl-mimic":
            mix = mix[:((mix.shape[0])//256)*256]
            sources = [source[:((mix.shape[0])//256)*256] for source in sources]


        mix_duration_ = mix.shape[0] / samplerate

        # Stack the sources in a tensor 
        sources_stacked_tensor = torch.stack(sources)

        # Do the separation using the function and get the overall performance
        estimate_sources, start_time_, end_time_, memory_ = model_separation_function(mix.unsqueeze(0))

        #TODO: Evaluate the separation for a model that's separating a mix with more files
        # than it's intended ones
        n_estimations = len(estimate_sources)
        # For each estimation, try to pair it to all the possible combinations
        # of sources, the one that gets the greatest average SIR, will be considered as
        # the original one
        if len(sources_stacked_tensor) > n_estimations:
            # All the possible combinations
            max_average_sir = -1e9
            max_average_sdr = -1e9
            closest_to_estimate_sources = None
            
            indexed_sources = [(i,source) for i,source in enumerate(sources)]
            for subset in itertools.combinations(indexed_sources,n_estimations):
                #Get the sources values as a numpy array
                subset_indexed_sources = sorted(subset, key = lambda x:x[0])
                sources_array = np.array([source[1].numpy() for source in sorted(subset_indexed_sources)])
                
                sdr_, sir_, sar_, perm = mir_eval.separation.bss_eval_sources(
                    reference_sources=sources_array,
                    estimated_sources=estimate_sources.numpy(),
                )

                # The one with the best average SIR (or SDR if only one estimation)
                # is the one we'll consider
                avg_sir = np.mean(sir_)
                avg_sdr = np.mean(sdr_)

                if (n_estimations > 1 and avg_sir > max_average_sir) or (n_estimations == 1 and avg_sdr > max_average_sdr): 
                    closest_to_estimate_sources = subset_indexed_sources
                    sdr_array = sdr_
                    sir_array = sir_
                    sar_array = sar_
                    max_average_sir = avg_sir
                    max_average_sdr = avg_sdr

            for j in range(n_estimations):
                j_val = closest_to_estimate_sources[perm[j]][0] # The source closest to the estimation
                mix_wav.append(mix_file)
                mix_duration.append(mix_duration_)
                original_source.append(source_files[j_val])
                if source_files[j_val].split("_")[-1] == mix_file.split("_")[-1]: 
                    main_source.append(True)
                else:
                    main_source.append(False)
                memory.append(memory_)
                start_time.append(start_time_)
                end_time.append(end_time_)
                sir.append(sir_array[j])
                sar.append(sar_array[j])
                sdr.append(sdr_array[j])
            continue

        # Evaluate the separation with mir_eval
        sdr_, sir_, sar_, perm = mir_eval.separation.bss_eval_sources(
            reference_sources=sources_stacked_tensor.numpy(),
            estimated_sources=estimate_sources.numpy(),
            )
        
        # "estimated source number perm[j] corresponds to true source number j"
        # Store the results

        for j in range(n_estimations):
            mix_wav.append(mix_file)
            mix_duration.append(mix_duration_)
            original_source.append(source_files[j])

            if source_files[j].split("_")[-1] == mix_file.split("_")[-1]: 
                main_source.append(True)
            else:
                main_source.append(False)
            memory.append(memory_)
            start_time.append(start_time_)
            end_time.append(end_time_)
            sir.append(sir_[j])
            sar.append(sar_[j])
            sdr.append(sdr_[j])
            # TODO: Add option to save the estimated sources
            # estimated source = estimate_sources[perm[j]]
    

    white_noise_snr_low = None
    white_noise_snr_high = None
    for operation in data_hparams["mixAugmentationPipeline"]:
        if operation["method"] == "addNoise":
            added_noise = True
            white_noise_snr_low = operation["snrLow"]
            white_noise_snr_high = operation["snrHigh"]
    df_results = pd.DataFrame(
        {
            "mix_file": mix_wav,
            "mix_duration": mix_duration,
            "mix_samplerate": [data_hparams["samplerate"] for i in range(len(mix_wav))],
            "num_speakers_in_mix": [data_hparams["numSpeakers"] for i in range(len(mix_wav))],
            "mix_snr_low": [data_hparams["mixSNRLow"] if data_hparams != 1 else None for i in range(len(mix_wav))],
            "mix_snr_high": [data_hparams["mixSNRHigh"] if data_hparams != 1 else None for i in range(len(mix_wav))],
            "white_noise_snr_low": [white_noise_snr_low for i in range(len(mix_wav))],
            "white_noise_snr_high": [white_noise_snr_high for i in range(len(mix_wav))],
            "original_source": original_source,
            "main_source":main_source,
            "start_time": start_time,
            "end_time":  end_time,
            "occupied_memory": memory,
            "SIR": sir,
            "SDR": sdr,
            "SAR": sar,

        }
    )

    return df_results
