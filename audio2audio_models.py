import time
from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.pretrained import WaveformEnhancement
from speechbrain.pretrained import SpectralMaskEnhancement


class Audio2AudioModels:
    def __init__(self):
        self.models = {
            "enhancers": {
                "mtl_mimic": WaveformEnhancement.from_hparams( 
                    source="speechbrain/mtl-mimic-voicebank",
                    savedir="pretrained_models/mtl-mimic-voicebank"),
                "metricgan": SpectralMaskEnhancement.from_hparams(
                    source="speechbrain/metricgan-plus-voicebank",
                    savedir="pretrained_models/metricgan-plus-voicebank")
            },
            "2speakers": {
                "resepformer": separator.from_hparams(
                    source="speechbrain/resepformer-wsj02mix", 
                    savedir='pretrained_models/resepformer-wsj02mix'),
                "sepformer": separator.from_hparams(
                    source="speechbrain/sepformer-wsj02mix", 
                    savedir='pretrained_models/sepformer-wsj02mix')
            },
            "3speakers": {
                "sepformer": separator.from_hparams(
                    source="speechbrain/sepformer-wsj03mix", 
                    savedir='pretrained_models/sepformer-wsj03mix')
            }
        }

    def enhancer_template(self,noisy_batch: torch.tensor, enhancement_function):
        lengths = torch.tensor([1.])
        st = time.time()
        estimate_source = enhancement_function(noisy_batch, lengths)
        et = time.time()
        elapsed_time = et - st
        memory = None
        return estimate_source, elapsed_time, memory
    
    def separator_template(self,mix_batch: torch.tensor, separator_function):
        st = time.time()
        estimate_sources = separator_function(mix=mix_batch)
        et = time.time()
        elapsed_time = et - st
        memory = None
        # Transform shape of estimate_sources
        estimate_sources = torch.transpose(estimate_sources,1,2)[0]
        return estimate_sources, elapsed_time, memory

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