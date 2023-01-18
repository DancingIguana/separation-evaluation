from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.pretrained import WaveformEnhancement
from speechbrain.pretrained import SpectralMaskEnhancement

# Saving enhanced signal on disk
torchaudio.save('enhanced.wav', enhanced.unsqueeze(0).cpu(), 16000)

# Enhancement = 1-speaker
#16K
# https://huggingface.co/speechbrain/mtl-mimic-voicebank
enhance_model = WaveformEnhancement.from_hparams( 
    source="speechbrain/mtl-mimic-voicebank",
    savedir="pretrained_models/mtl-mimic-voicebank",
)
# https://huggingface.co/speechbrain/metricgan-plus-voicebank
enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)


# 2-speaker
#8K
#https://huggingface.co/speechbrain/resepformer-wsj02mix
model = separator.from_hparams(
    source="speechbrain/resepformer-wsj02mix", 
    savedir='pretrained_models/resepformer-wsj02mix'
)
#https://huggingface.co/speechbrain/sepformer-wsj02mix
model = separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix", 
    savedir='pretrained_models/sepformer-wsj02mix')
