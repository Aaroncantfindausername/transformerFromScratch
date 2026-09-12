from pathlib import Path
from transformerfromscratch.models.transformer import (
    TinyGPT,
)
import torch
import numpy as np

from transformerfromscratch.pipeline.data import (
    get_project_root,
    load_data,
)
from transformerfromscratch.tokeniser.tokeniser import BasicTokeniser


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
project_root = get_project_root()
config = torch.load(project_root / "models" / "model_config.pth")
model_state_dict = torch.load(project_root / "models" / "model.pth")
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

dataset_path = project_root / "datasets" / "tiny_shakespear.txt"
text = load_data(dataset_path)
tokeniser = BasicTokeniser(300)
tokeniser.train(text)
encoded_text = tokeniser.encode(text[:200])
decoded_text = tokeniser.decode(encoded_text)
print(encoded_text)
print(decoded_text)
