"""
Map a model checkpoint with its pythonic model interface to enable global cli
"""
from pathlib import Path
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.utils.checks import check_file

from ultralytics.yolo.engine.model import YOLO
from ultralytics.vit.rtdetr.model import RTDETR
from ultralytics.nn.modules.head import Detect, Segment, Classify, Pose, RTDETRDecoder

MODEL_REGISTRY = {
  "YOLO": [Detect, Segment, Classify, Pose],
  "RTDETR": [RTDETRDecoder],
  "SAM": None, # TODO: SAM heads
}

def _load(self, weights: str):
    """
    Initializes a new model and infers the task type from the model head.

    Args:
        weights (str): model checkpoint to be loaded
        task (str) or (None): model task
    """
    suffix = Path(weights).suffix
    if suffix == '.pt':
        model, ckpt = attempt_load_one_weight(weights)
    else:
        weights = check_file(weights)
        model, ckpt = weights, None
    
    return model, ckpt

def get_model_interface(weights):
    model, ckpt = _load(weights)
    head = None # get head from the model 
    for base_model, heads in MODEL_REGISTRY.items():
        if head in heads:
            return eval(base_model)(ckpt)
    
    raise ModuleNotFoundError("Couldn't identify model checkpoint. Please try loading it manually")

