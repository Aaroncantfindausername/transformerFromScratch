import pickle as pkl
import torch
from transformerfromscratch.pipeline.data import (
    get_project_root,
    load_data,
)
from transformerfromscratch.tokeniser.tokeniser import BasicTokeniser

config = {
    "vocab_size": 1024,
}
project_root = get_project_root()
model_state_dict = torch.load(project_root / "models" / "model.pth")
dataset_path = project_root / "datasets" / "tiny_shakespear.txt"
text = load_data(dataset_path)
tokeniser = BasicTokeniser(config["vocab_size"])
tokeniser.train(text)
with open(project_root / "models" / "tokeniser.pkl", "wb") as f:
    pkl.dump(tokeniser, f)
