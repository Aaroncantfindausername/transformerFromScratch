from torch.nn.modules import CrossEntropyLoss
import torch
from torch import optim
from transformerfromscratch.models.transformer import (
    MultiHeadedSelfAttention,
    SelfAttention,
)
from transformerfromscratch.pipeline.data import (
    encode_text,
    load_data,
    train_val_test_split,
)
from transformerfromscratch.pipeline.train import train_full

config = {
    "seed": 67,
    "context_length": 8,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 1000,
    "early_stop_patience": 20,
    "scheduler_patience": 7,
}
text = load_data("datasets/tiny_shakespear.txt")
encoded_text, decoder = encode_text(text)
train_loader, val_loader, test_loader = train_val_test_split(
    encoded_text,
    context_length=config["context_length"],
    batch_size=config["batch_size"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = MultiHeadedSelfAttention(
    config["d_model"],
    config["d_q"],
    config["seq_len"],
    config["batch_size"],
    config["dropout"],
    config["n_heads"],
    device,
)
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
