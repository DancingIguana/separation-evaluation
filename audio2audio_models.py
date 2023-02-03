import time
from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.pretrained import WaveformEnhancement
from speechbrain.pretrained import SpectralMaskEnhancement
import torch
import psutil
import os

class Audio2AudioModels:
    """
    Class to call an enhancer or separator function from pretrained 
    Speechbrain's models given its type and model name.

    When running any of the models, the output will consist of the following:
    - estimate_source(s): the separated or cleaned signals obtained by the 
        model.
    - time: the time it took to run the model given the specific signal batch 
        in seconds.
    - memory: the amount of memory that was being used when running the model 
        in bytes.
    
    Parameters:
    ---------------------
    - model_type: the type of the model (enhancers, 2speakers, 3speakers).
    - model_name: the specific name of the model, check audio2audio_models.py
        for seeing the available models.
    """
    def __init__(self, model_type:str, model_name: str):
        assert(model_type in ["enhancers","2speakers","3speakers"], f"Model type {model_type} doesn't exist")
        
        self.model_type = model_type
        self.model_name = model_name
        self.models = {
            "enhancers": {
                "mtl_mimic": WaveformEnhancement.from_hparams( 
                    source="speechbrain/mtl-mimic-voicebank",
                    savedir="pretrained_models/mtl-mimic-voicebank"),
                "metricgan": SpectralMaskEnhancement.from_hparams(
                    source="speechbrain/metricgan-plus-voicebank",
                    savedir="pretrained_models/metricgan-plus-voicebank"),
                "sepformer-wham16k": separator.from_hparams(
                    source="speechbrain/sepformer-wham16k-enhancement", 
                    savedir='pretrained_models/sepformer-wham16k-enhancement')
            },
            "2speakers": {
                "resepformer-wsj02mix": separator.from_hparams(
                    source="speechbrain/resepformer-wsj02mix", 
                    savedir='pretrained_models/resepformer-wsj02mix'),
                "sepformer-wsj02mix": separator.from_hparams(
                    source="speechbrain/sepformer-wsj02mix", 
                    savedir='pretrained_models/sepformer-wsj02mix')
            },
            "3speakers": {
                "sepformer-wsj03mix": separator.from_hparams(
                    source="speechbrain/sepformer-wsj03mix", 
                    savedir='pretrained_models/sepformer-wsj03mix')
            }
        }

        self.model = self.models[model_type][model_name]

    def enhancer_template(self,noisy_batch: torch.tensor, enhancement_function):
        """
        Function for most of the enhancers from Speechbrain.
        """
        lengths = torch.tensor([1.])
        this_process = psutil.Process(os.getpid())
        st = time.time()
        #print("Noisy batch size",noisy_batch.shape)
        estimate_source = enhancement_function(noisy_batch, lengths)
        #print("Clean batch size",estimate_source.shape)
        et = time.time()
        elapsed_time = et - st
        memory = this_process.memory_info().rss
        return estimate_source, elapsed_time, memory
    
    def separator_template(self,mix_batch: torch.tensor, separator_function):
        """
        Function for the separators from Speechbrain.
        """
        this_process = psutil.Process(os.getpid())
        st = time.time()
        estimate_sources = separator_function(mix=mix_batch)
        et = time.time()
        elapsed_time = et - st
        memory = this_process.memory_info().rss
        # Transform shape of estimate_sources
        estimate_sources = torch.transpose(estimate_sources,1,2)[0]
        return estimate_sources, elapsed_time, memory


    def audio_model_function(self,noisy_batch: torch.tensor):
        """
        Function to be called when using the class

        Returns:
        -------------------
        - estimated_source(s) given by the Speechbrain model.
        - time it took to run the Speechbrain model with the batch in seconds.
        - the memory that was being used while running the function in bytes.
        """
        model_function = None
        
        if self.model_name == "mtl_mimic":
            noisy_bacth = noisy_batch[:,:(noisy_batch.shape[1]//256)*256]

        if self.model_type == "enhancers" and "sepformer" not in self.model_name:
            model_function = self.models[self.model_type][self.model_name].enhance_batch
            return self.enhancer_template(
                noisy_batch=noisy_batch,
                enhancement_function=model_function
            )
        
        model_function = self.models[self.model_type][self.model_name].separate_batch
        return self.separator_template(
            mix_batch=noisy_batch,
            separator_function=model_function
        )