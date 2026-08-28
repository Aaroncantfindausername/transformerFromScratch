from pathlib import Path
from transformerfromscratch.models.transformer import (
    TinyGPT,
)
import torch
import numpy as np

from transformerfromscratch.pipeline.data import (
    decode_to_text,
    encode_text,
    get_encoder_decoder,
    get_project_root,
    load_data,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
project_root = get_project_root()
config = torch.load(project_root / "models" / "model_config.pth")
model_state_dict = torch.load(project_root / "models" / "model.pth")
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

dataset_path = project_root / "datasets" / "tiny_shakespear.txt"
text = load_data(dataset_path)
encoded_text, encoder, decoder = get_encoder_decoder(text)
model = TinyGPT(
    config["d_model"],
    config["d_q"],
    config["dropout"],
    config["context_window"],
    config["batch_size"],
    config["mlp_layers"],
    config["mlp_activation"],
    config["n_attn_heads"],
    config["n_blocks"],
    len(decoder),
    device,
)
model.load_state_dict(model_state_dict)
model.eval()
x = torch.tensor(encoded_text[:32], device=device).unsqueeze(0)
output = model.generate(x, 1000)
print(decode_to_text(output.squeeze().tolist(), decoder))
