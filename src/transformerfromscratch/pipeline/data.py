from typing import Tuple
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader


def load_data(file: str) -> str:
    with open(file) as f:
        text = f.read()
    return text


def encode_text(text: str) -> Tuple[list[int], dict[int, str]]:
    chars = sorted(list(set(text)))
    char_to_i = {c: i for i, c in enumerate(chars)}
    i_to_char = {i: c for i, c in enumerate(chars)}
    ints = [char_to_i[c] for c in text]
    return ints, i_to_char


def decode_to_text(ints: list[int], dict: dict[int, str]) -> str:
    return "".join([dict[i] for i in ints])


def train_val_test_split(
    encoded_text: list[int],
    context_length: int,
    batch_size: int,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    X = []
    y = []
    for i in range(len(encoded_text) - context_length - 1):
        X.append(encoded_text[i : i + context_length])
        y.append(encoded_text[i + 1 : i + context_length + 1])
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, train_size=1 - val_split - test_split, shuffle=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=test_split / (test_split + val_split),
        shuffle=True,
    )
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size, True)
    val_loader = DataLoader(val_dataset, batch_size, False)
    test_loader = DataLoader(test_dataset, batch_size, False)
    return (train_loader, val_loader, test_loader)
