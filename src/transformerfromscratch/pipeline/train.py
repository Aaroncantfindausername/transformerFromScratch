from typing import Any
import copy
import torch
from torch.utils.data import DataLoader


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
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


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion,
    device: str | torch.device,
) -> float:
    model.eval()
    running_loss: float = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = (
                inputs.to(device),
                targets.to(device),
            )
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    avg_loss: float = running_loss / len(loader.dataset)
    return avg_loss


def train_full(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: str | torch.device,
    epochs: int,
    patience: int,
    load_best_val_loss: bool = True,
):
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.10f} , Val loss: {val_loss:.10f}, "
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve > patience:
            print(
                f"Validation loss not improved for more than {patience} epochs, early stop triggered"
            )
            break
    if load_best_val_loss:
        model.load_state_dict(best_model_wts)
