import pandas as pd
import json
from speechbrain.dataio.dataio import read_audio
import mir_eval
import torch
import numpy as np

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
        print("Sources shape",sources_stacked_tensor.shape)
        print("Mix shape",mix.shape)
        # Do the separation using the function and get the overall performance
        estimate_sources, time_, memory_ = model_separation_function(mix.unsqueeze(0))
        print("Estimate shape", estimate_sources.shape)

        #TODO: Evaluate the separation for a model that's separating a mix with more files
        # than it's intended ones
        n_estimations = len(estimate_sources)
        if len(sources_stacked_tensor) > n_estimations:
            # Simply add "dummy" estimations and we care only of the original estimations
            remaining_sources = len(sources) - n_estimations
            estimate_sources = torch.cat(
                (
                    estimate_sources,
                    torch.zeros((remaining_sources,estimate_sources.shape[1]))
                )
            )

            # The top estimations are the ones we store
            # Evaluate the separation with mir_eval
            sdr_, sir_, sar_, perm = mir_eval.separation.bss_eval_sources(
                reference_sources=sources_stacked_tensor.numpy(),
                estimated_sources=estimate_sources.numpy(),
                )
            # One tuple represents one estimation's score
            results = [(j,sdr_[j],sir_[j],sar_[j]) for j in range(len(estimate_sources))]
            #Sort the values from greatest to lowest SIR
            results = sorted(results, key = lambda x : x[2], reverse=True)
            
            # Now store only the best results:
            for j in range(n_estimations):
                j_val, sdr_val, sir_val, sar_val = results[j]
                mix_wav.append(mix_file)
                mix_duration.append(mix_duration_)
                original_source.append(source_files[j])

                if source_files[j_val].split("_")[-1] == mix_file.split("_")[-1]: 
                    main_source.append(True)
                else:
                    main_source.append(False)
                memory.append(memory_)
                time.append(time_)
                sir.append(sir_val)
                sar.append(sar_val)
                sdr.append(sdr_val)
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