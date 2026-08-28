from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F


def get_project_root() -> Path:
    # 1. Determine the starting directory
    try:
        # Works when running as a standard Python script
        start_path = Path(__file__).resolve().parent
    except NameError:
        # Fallback for IPython / Jupyter Notebooks
        start_path = Path.cwd()

    # 2. Traverse up the tree to find the root marker
    # You can change "pyproject.toml" to ".git" or a custom ".root" file
    for parent in start_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback if no marker is found (e.g., in a Docker container without .git)
    return start_path


project_root = get_project_root()
config = torch.load(project_root / "models" / "model_config.pth")


def load_data(file: str | Path) -> str:
    with open(file) as f:
        text = f.read()
    return text


def get_encoder_decoder(
    text: str,
) -> Tuple[list[int], dict[str, int], dict[int, str]]:
    chars = sorted(list(set(text)))
    char_to_i = {c: i for i, c in enumerate(chars)}
    i_to_char = {i: c for i, c in enumerate(chars)}
    ints = [char_to_i[c] for c in text]
    return ints, char_to_i, i_to_char


def decode_to_text(ints: list[int], dict: dict[int, str]) -> str:
    return "".join([dict[i] for i in ints])


def encode_text(text: str, dict: dict[str, int]) -> list[int]:
    return [dict[c] for c in text]


def train_val_test_split(
    encoded_text: list[int],
    context_window: int,
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
    for i in range(len(encoded_text) - context_window - 1):
        X.append(encoded_text[i : i + context_window])
        y.append(encoded_text[i + 1 : i + context_window + 1])
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
