"""
lambada.py
Modern LAMBADA Evaluation Pipeline
PyTorch + Transformers
"""

import json
import requests
import ftfy
import numpy as np

from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

LAMBADA_URL = (
    "https://openaipublic.blob.core.windows.net/"
    "gpt-2/data/lambada_test.jsonl"
)


class LambadaDataset:

    def __init__(
        self,
        model_name="gpt2",
        cache_file="lambada.json"
    ):

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.cache_file = Path(
            cache_file
        )

    def download(self):

        response = requests.get(
            LAMBADA_URL,
            timeout=30
        )

        response.raise_for_status()

        texts = []

        for line in response.text.splitlines():

            item = json.loads(line)

            text = ftfy.fix_text(
                item["text"]
            )

            texts.append(text)

        return texts

    def tokenize(self, texts):

        tokenized = []

        for text in texts:

            ids = self.tokenizer.encode(
                text
            )

            tokenized.append(ids)

        return tokenized

    def load(self):

        if self.cache_file.exists():

            with open(
                self.cache_file,
                "r",
                encoding="utf-8"
            ) as f:

                return json.load(f)

        texts = self.download()

        tokens = self.tokenize(
            texts
        )

        with open(
            self.cache_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                tokens,
                f
            )

        return tokens

    def create_dataset(
        self,
        max_length=1024
    ):

        token_lists = self.load()

        samples = []

        for tokens in token_lists:

            if len(tokens) > max_length:

                tokens = tokens[
                    :max_length
                ]

            samples.append(
                {
                    "input_ids": tokens
                }
            )

        return Dataset.from_list(
            samples
        )


def compute_perplexity(
    model,
    tokenizer,
    text,
    device
):

    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        outputs = model(
            **inputs,
            labels=inputs["input_ids"]
        )

    loss = outputs.loss

    perplexity = torch.exp(
        loss
    )

    return perplexity.item()


if __name__ == "__main__":

    dataset_builder = LambadaDataset()

    dataset = (
        dataset_builder
        .create_dataset()
    )

    print(
        f"Dataset Size: "
        f"{len(dataset)}"
    )

    print(
        dataset[0]
    ) 