"""
utils.py
Modern Utility Functions
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List

import torch

# Logging

def setup_logging(log_name: str):

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_dir / f"{log_name}.log"
    )

    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



# Batch Size


def get_batch_size(config: Dict):

    mode = config["mode"]

    return config.get(
        f"{mode}_batch_size",
        config.get("batch_size", 1)
    )



# Training Mode

def set_mode(config: Dict, mode: str):

    valid_modes = {
        "train",
        "eval",
        "predict"
    }

    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode: {mode}"
        )

    config["mode"] = mode

    return config


# Config Save

def save_config(
    config: Dict,
    output_dir: str
):

    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    config_file = (
        output_path / "config.json"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    print(
        f"Config saved: {config_file}"
    )

# Delete Directory


def remove_directory(path):

    path = Path(path)

    if path.exists():

        shutil.rmtree(path)

        print(
            f"Deleted: {path}"
        )

# Expand Attention Type

def expand_attention_types(
    config_list
):

    output = []

    for item, repeat in config_list:

        for _ in range(repeat):

            output.extend(item)

    return output



# Count Parameters


def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable Parameters:"
        f"\n{total:,}\n"
    )

    return total



# Model Summary


def print_model_info(model):

    count_parameters(model)

    print("\nLayers:\n")

    for name, module in model.named_modules():

        if name:

            print(name)



# Loss Denominator

def loss_denominator(
    batch_size,
    sequence_length
):

    return float(
        batch_size * sequence_length
    )



# Dataset Preview


def preview_dataset(
    dataset,
    tokenizer,
    samples=1
):

    print("-" * 80)

    for i in range(samples):

        text = tokenizer.decode(
            dataset[i]["input_ids"]
        )

        print(text[:500])

        print("\n...\n")

    print("-" * 80)


# Auto Device


def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

# Natural Sort


def natural_sort(items: List[str]):

    def convert(text):

        return (
            int(text)
            if text.isdigit()
            else text.lower()
        )

    def key(item):

        return [
            convert(c)
            for c in re.split(
                r"([0-9]+)",
                item
            )
        ]

    return sorted(
        items,
        key=key
    ) 