"""
Modern GPT-Neo Style Test Suite
PyTorch + HuggingFace + PyTest
"""

import pytest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokeniz

# Fixtures


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return model

# Forward Pass Test


def test_model_forward(model, tokenizer):

    text = "Hello GPT Neo"

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits is not None
    assert outputs.logits.shape[0] == 1

    print("Forward Pass Test Passed")



# Text Generation Test


def test_text_generation(model, tokenizer):

    prompt = "Artificial Intelligence"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        generated = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_p=0.95
        )

    assert generated.shape[1] > inputs["input_ids"].shape[1]

    generated_text = tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )

    print("\nGenerated Text:")
    print(generated_text)

    print("Generation Test Passed")


# Loss Computation Test


def test_loss_computation(model, tokenizer):

    text = "Machine Learning is amazing"

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    outputs = model(
        input_ids=inputs["input_ids"],
        labels=inputs["input_ids"]
    )

    loss = outputs.loss

    assert loss is not None
    assert loss.item() > 0

    print(f"Loss = {loss.item():.4f}")

    print("Loss Test Passed")


# MLM Masking Test


def mlm_mask_tokens(
        input_ids,
        mask_token_id=103,
        mask_prob=0.15
):

    labels = input_ids.clone()

    probability_matrix = torch.full(
        labels.shape,
        mask_prob
    )

    masked_indices = torch.bernoulli(
        probability_matrix
    ).bool()

    input_ids[masked_indices] = mask_token_id

    return input_ids, labels


def test_mlm_masking():

    input_ids = torch.tensor(
        [10, 20, 30, 40, 50]
    )

    masked_ids, labels = mlm_mask_tokens(
        input_ids
    )

    assert masked_ids.shape == labels.shape

    print("MLM Test Passed")


# Sampling Test


def test_sampling():

    logits = torch.randn(
        1,
        1000
    )

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    assert token.shape == (1, 1)

    print("Sampling Test Passed")

# Entmax Test


def test_softmax_distribution():

    x = torch.randn(10)

    y = torch.softmax(
        x,
        dim=-1
    )

    assert torch.isclose(
        y.sum(),
        torch.tensor(1.0),
        atol=1e-5
    )

    print("Distribution Test Passed")



# Main Runner

if __name__ == "__main__":

    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2"
    )

    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        "gpt2"
    )

    test_model_forward(
        model,
        tokenizer
    )

    test_text_generation(
        model,
        tokenizer
    )

    test_loss_computation(
        model,
        tokenizer
    )

    test_mlm_masking()

    test_sampling()

    test_softmax_distribution()

    print("\nAll Tests Passed Successfully") 