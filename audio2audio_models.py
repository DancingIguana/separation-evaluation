import time
from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.pretrained import WaveformEnhancement
from speechbrain.pretrained import SpectralMaskEnhancement
import torch
import psutil
import os
import json

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
    """
    def __init__(self):
        self.model_type = None
        self.model_name = None        
        self.model = None

        with open("./project_variables.json", "r") as f:
            project_variables = json.load(f)
        
        self.models = project_variables["models"]

    def enhancer_template(self,noisy_batch: torch.tensor, enhancement_function):
        """
        Function for most of the enhancers from Speechbrain.
        """
        lengths = torch.tensor([1.])
        this_process = psutil.Process(os.getpid())
        st = time.time()
        estimate_source = enhancement_function(noisy_batch, lengths)
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
        assert(self.model != None)
        model_function = None
        
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

    def load_model(self, model_type: str, model_name: str):
        assert(model_type in self.models and model_name in self.models[model_type])
        self.model_type = model_type
        self.model_name = model_name
        if model_type == "enhancers":
            if model_name == "mtl-mimic":
                self.model = WaveformEnhancement.from_hparams( 
                    source="speechbrain/mtl-mimic-voicebank",
                    savedir="pretrained_models/mtl-mimic-voicebank"
                )
            elif model_name == "metricgan":
                self.model = SpectralMaskEnhancement.from_hparams(
                    source="speechbrain/metricgan-plus-voicebank",
                    savedir="pretrained_models/metricgan-plus-voicebank")

            elif model_name == "sepformer-wham16k":
                self.model = separator.from_hparams(
                    source="speechbrain/sepformer-wham16k-enhancement", 
                    savedir='pretrained_models/sepformer-wham16k-enhancement'
                )
        elif model_type == "2speakers":
            if model_name == "resepformer":
                self.model = separator.from_hparams(
                    source="speechbrain/resepformer-wsj02mix", 
                    savedir='pretrained_models/resepformer-wsj02mix'
                )
            elif model_name == "sepformer":
                self.model = separator.from_hparams(
                    source="speechbrain/sepformer-wsj02mix", 
                    savedir='pretrained_models/sepformer-wsj02mix')

        elif model_type == "3speakers":
            if model_name == "sepformer":
                self.model = separator.from_hparams(
                    source="speechbrain/sepformer-wsj03mix", 
                    savedir='pretrained_models/sepformer-wsj03mix')