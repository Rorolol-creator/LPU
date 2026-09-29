import numpy as np
import torch
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

torch.set_num_threads(1)

_weights = MobileNet_V2_Weights.IMAGENET1K_V1
_model = mobilenet_v2(weights=_weights)
_model.classifier = torch.nn.Identity()
_model.eval()
_preprocess = _weights.transforms()


def pretreat(image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    tensor = _preprocess(tensor)
    with torch.no_grad():
        features = _model(tensor.unsqueeze(0))
    return features.squeeze(0).numpy()
