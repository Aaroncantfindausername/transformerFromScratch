from torch.nn.modules import CrossEntropyLoss
import torch
from torch import optim
from transformerfromscratch.models.transformer import (
    TinyGPT,
)
from transformerfromscratch.pipeline.data import (
    get_encoder_decoder,
    get_project_root,
    load_data,
    train_val_test_split,
)
from transformerfromscratch.pipeline.train import train_full


config = {
    "seed": 67,
    "d_model": 64,
    "d_q": 32,
    "dropout": 0.3,
    "context_window": 32,
    "batch_size": 1024,
    "mlp_layers": 3,
    "mlp_activation": "relu",
    "n_attn_heads": 4,
    "n_blocks": 4,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "epochs": 200,
    "early_stop_patience": 20,
    "scheduler_patience": 7,
}
project_root = get_project_root()
dataset_path = project_root / "datasets" / "tiny_shakespear.txt"
text = load_data(dataset_path)
encoded_text, _, decoder = get_encoder_decoder(text)
train_loader, val_loader, test_loader = train_val_test_split(
    encoded_text,
    context_window=config["context_window"],
    batch_size=config["batch_size"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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
torch.compile(model)
criterion = CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=config["scheduler_patience"],
    factor=0.5,
    threshold=0.0,
)
train_full(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    config["epochs"],
    config["early_stop_patience"],
    load_best_val_loss=True,
)
torch.save(model.state_dict(), project_root / "models" / "model.pth")
torch.save(config, project_root / "models" / "model_config.pth")
