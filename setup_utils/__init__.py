import sys
sys.path.append("../separation-evaluation/")
sys.path.append("../separation-evaluation/speechbrain") # Adds higher directory to python modules path.
from setup_minilibrispeech import setup_minilibrispeech
from setup_models import setup_models
from setup_mix_hparams import setup_mix_hparams
from setup_evaluation_hparams import setup_evaluation_hparams