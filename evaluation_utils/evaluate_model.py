import pandas as pd
import json
from speechbrain.dataio.dataio import read_audio
import mir_eval

def evaluate_model(
    dataset_hparams_json: str,
    num_speakers: int, 
    model_separation_function, 
    samplerate: int = 16000
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
        data_hparams = json.load(dataset_hparams_json)

    dataset_df = pd.read_csv(dataset_hparms_json)
    mix_wav = []
    original_source = []
    mix_duration = []
    time = []
    sdr = []
    sir = []
    sar = []
    memory = []

    for index, row in dataset_df.iterrows():

        # Get the mix file and the corresponding sources
        mix_file = row["mix_wav"]
        source_files = [row[f"s{i+1}_wav"] for i in range(num_speakers)]
        
        mix = read_audio(mix_file)
        sources = [read_audio(source_file) for source_file in source_files]

        mix_duration_ = mix.shape[0] / samplerate

        # Stack the sources in a tensor 
        sources_stacked_tensor = torch.stack(sources)

        # Do the separation using the function and get the overall performance
        estimate_sources, time_, memory_ = model_separation_function(mix.unsqueeze(0))

        # Evaluate the separation with mir_eval
        sdr_, sir_, sar_, perm = mir_eval.separation.bss_eval_sources(
            reference_sources=sources_stacked_tensor.numpy(),
            estimated_sources=estimate_sources.numpy(),
            )
        
        # "estimated source number perm[j] corresponds to true source number j"
        # Store the results
        for j in range(len(perm)):
            mix_wav.append(mix_file)
            mix_duration.append(mix_duration_)
            original_source.append(source_files[j])
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
            "separation_time": time,
            "occupied_memory": memory,
            "SIR": sir,
            "SDR": sdr,
            "SAR": sar
        }
    )

    return df_results