from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.pretrained import WaveformEnhancement
from speechbrain.pretrained import SpectralMaskEnhancement
from audio2audio_models import Audio2AudioModels

def setup_models(project_variables_file: str) -> None:
    """
    Given a dictionary of lists of models, load the ones
    that are available under Audio2AudioModels class
    """
    with open(project_variables_file, "r") as f:
        project_variables = json.load(f)
    models = project_variables["models"]
    model_class = Audio2AudioModels()
    for model_type in models:
        for model_name in model_type:
            model_class.load_model(
                model_type = model_type, 
                model_name = model_name
            )


