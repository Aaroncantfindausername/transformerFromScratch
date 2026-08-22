from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion,
    optimizer,
    device: str | torch.device,
    classification: bool = False,
) -> float:
    batch_loss = None
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        batch_loss = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += batch_loss * inputs.size(0)
    return running_loss / len(loader.dataset)
