import pandas as pd
import json
from speechbrain.dataio.dataio import read_audio
import mir_eval
import torch
import numpy as np
import itertools

def evaluate_model(
    dataset_hparams_json: str,
    model_separation_function, 
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
    - dataset_df: the dataframe containing a column of the mix file ('mix_wav') 
    and a series of columns containing the source files for said mix ('si_wav')
    - num_speakers: the number of sources that are in the mix
    - model_separation_function: the function that given the mix returns the
    estimated sources, time of execution and occupied memory
    - samplerate: the samplerate of the mixes.

    Returns:
    ----------------------
    Dataframe with the following columns:
    - mix_file: path of the original mix
    - mix_duration: duration in seconds of the original mix
    - original_source: path of the original source that was separated
    - separation_time: time it took to run the separation function
    - occupied_memory: the memory that the process of separation took
    - SIR: separation quality variable
    - SAR: separation quality variable
    - SDR: separation quality variable
    """
    with open(dataset_hparams_json,"r") as f:
        data_hparams = json.load(f)

    num_speakers = data_hparams["numSpeakers"]
    samplerate = data_hparams["samplerate"]
    dataset_df = pd.read_csv(data_hparams["csv"])
    mix_wav = []
    original_source = []
    mix_duration = []
    time = []
    sdr = []
    sir = []
    sar = []
    memory = []
    main_source = []
    for index, row in dataset_df.iterrows():

        # Get the mix file and the corresponding sources
        mix_file = row["mix_wav"]
        source_files = [row[f"s{i+1}_wav"] for i in range(num_speakers)]
        
        mix = read_audio(mix_file)
        sources = [read_audio(source_file) for source_file in source_files]

        mix_duration_ = mix.shape[0] / samplerate

        # Stack the sources in a tensor 
        sources_stacked_tensor = torch.stack(sources)
        #print("Sources shape",sources_stacked_tensor.shape)
        #print("Mix shape",mix.shape)
        # Do the separation using the function and get the overall performance
        estimate_sources, time_, memory_ = model_separation_function(mix.unsqueeze(0))
        #print("Estimate shape", estimate_sources.shape)

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
                print(f"Subset indexes sources: {subset_indexed_sources}")
                sources_array = np.array([source[1].numpy() for source in sorted(subset_indexed_sources)])
                print(f"Sources array: {sources_array}")
                sdr_, sir_, sar_, perm = mir_eval.separation.bss_eval_sources(
                    reference_sources=sources_array,
                    estimated_sources=estimate_sources.numpy(),
                )

                # The one with the best average SIR (or SDR if only one estimation)
                # is the one we'll consider
                avg_sir = np.mean(sir_)
                avg_sdr = np.mean(sdr_)
                #print(f"SIR: {sir_}")
                #print(f"SDR: {sdr_}")
                if (n_estimations > 1 and avg_sir > max_average_sir) or (n_estimations == 1 and avg_sdr > max_average_sdr): 
                    closest_to_estimate_sources = subset_indexed_sources
                    sdr_array = sdr_
                    sir_array = sir_
                    sar_array = sar_
                    max_average_sir = avg_sir
                    max_average_sdr = avg_sdr
                    #print("Max average",max_average_sdr)

            #print(f"Chosen:\n\t SIR:{sir_array}, \n\tSDR:{sdr_array} \n\tIndex: {[j[0] for j in subset_indexed_sources]}")
            for j in range(n_estimations):
                
                j_val = subset_indexed_sources[j][0]
                mix_wav.append(mix_file)
                mix_duration.append(mix_duration_)
                original_source.append(source_files[j_val])
                if source_files[j_val].split("_")[-1] == mix_file.split("_")[-1]: 
                    main_source.append(True)
                else:
                    main_source.append(False)
                memory.append(memory_)
                time.append(time_)
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
            time.append(time_)
            sir.append(sir_[j])
            sar.append(sar_[j])
            sdr.append(sdr_[j])
            # TODO: Add option to save the estimated sources
            # estimated source = estimate_sources[perm[j]]
        
    df_results = pd.DataFrame(
        {
            "mix_file": mix_wav,
            "mix_duration": mix_duration,
            "original_source": original_source,
            "main_source":main_source,
            "separation_time": time,
            "occupied_memory": memory,
            "SIR": sir,
            "SDR": sdr,
            "SAR": sar
        }
    )

    return df_results