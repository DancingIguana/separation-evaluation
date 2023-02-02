import time
from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.pretrained import WaveformEnhancement
from speechbrain.pretrained import SpectralMaskEnhancement
import torch
import psutil
import os

class Audio2AudioModels:
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
                "sepformer-whamr16k": separator.from_hparams(
                    source="speechbrain/sepformer-whamr16k", 
                    savedir='pretrained_models/sepformer-whamr16k')
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
        model_function = None
        
        if self.model_type == "enhancers" and self.model_name != "sepformer":
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

    def enhancer_mtl_mimic(self,noisy_batch:torch.tensor):
        enhancement_function = self.models["enhancers"]["mtl_mimic"].enhance_batch
        return self.enhancer_template(
            noisy_batch=noisy_batch,
            enhancement_function=enhancement_function)


    def enhancer_metricgan(self,noisy_batch:torch.tensor):
        enhancement_function = self.models["enhancers"]["metricgan"].enhance_batch
        return self.enhancer_template(
            noisy_batch=noisy_batch,
            enhancement_function=enhancement_function)


    def separator_resepformer_2speakers(self,mix_batch:torch.tensor):
        separator_function = self.models["2speakers"]["resepformer"].separate_batch
        return self.separator_template(
            mix_batch=mix_batch,
            separator_function=separator_function)


    def separator_sepformer_2speakers(self,mix_batch:torch.tensor):
        separator_function = self.models["2speakers"]["sepformer"].separate_batch
        return self.separator_template(
            mix_batch=mix_batch,
            separator_function=separator_function)


    def separator_sepformer_3speakers(self,mix_batch: torch.tensor):
        separator_function = self.models["3speakers"]["sepformer"].separate_batch
        return self.separator_template(
            mix_batch=mix_batch,
            separator_function=separator_function)