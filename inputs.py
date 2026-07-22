"""
dataset.py
Modern Dataset Pipeline
PyTorch + Transformers
"""

import random
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class GPTDataset(Dataset):

    def __init__(
        self,
        texts,
        tokenizer,
        max_length=1024
    ):

        self.examples = []

        for text in texts:

            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            self.examples.append(
                encoded["input_ids"].squeeze(0)
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        input_ids = self.examples[idx]

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }


class MLMDataset(Dataset):

    def __init__(
        self,
        texts,
        tokenizer,
        max_length=128,
        mask_probability=0.15
    ):

        self.examples = []
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability

        for text in texts:

            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            self.examples.append(
                encoded["input_ids"].squeeze(0)
            )

    def mask_tokens(
        self,
        input_ids
    ):

        labels = input_ids.clone()

        probability_matrix = torch.full(
            labels.shape,
            self.mask_probability
        )

        mask = torch.bernoulli(
            probability_matrix
        ).bool()

        input_ids[mask] = (
            self.tokenizer.mask_token_id
        )

        return input_ids, labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        input_ids = self.examples[idx].clone()

        masked_inputs, labels = self.mask_tokens(
            input_ids
        )

        return {
            "input_ids": masked_inputs,
            "labels": labels
        }


def generate_text(
    model,
    tokenizer,
    prompt,
    max_tokens=100,
    device="cpu"
):

    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2"
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    sample_texts = [

        "Artificial Intelligence is transforming industries.",

        "Machine Learning enables predictive systems.",

        "Large Language Models are powerful tools."
    ]

    dataset = GPTDataset(
        sample_texts,
        tokenizer,
        max_length=128
    )

    print(dataset[0])

    mlm_dataset = MLMDataset(
        sample_texts,
        tokenizer,
        max_length=128
    )

    print(mlm_dataset[0]) 