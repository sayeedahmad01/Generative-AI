import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel



# Dataset

class TextDataset(Dataset):

    def __init__(self, texts, tokenizer, max_length=128):
        self.examples = []

        for text in texts:
            tokens = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            self.examples.append(
                tokens["input_ids"].squeeze()
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]



# Training Function

def train_model(model, dataloader, device, epochs):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for batch in dataloader:

            batch = batch.to(device)

            outputs = model(
                input_ids=batch,
                labels=batch
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1} | Loss: {total_loss:.4f}"
        )



# Text Generation

def generate_text(model, tokenizer, prompt, device):

    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    print("\nGenerated Text:\n")
    print(text)

# Main Function

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        action="store_true"
    )

    parser.add_argument(
        "--predict",
        action="store_true"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Artificial Intelligence is"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using Device: {device}")

    # Load Saved Model
    tokenizer = GPT2Tokenizer.from_pretrained(
        "saved_model"
    )

    model = GPT2LMHeadModel.from_pretrained(
        "saved_model"
    )

    model.to(device)

    # TRAIN
    
    if args.train:

        texts = [
            "Sayeed Ahmad is a Computer Science Engineer.",
            "Sayeed Ahmad builds AI and Machine Learning projects.",
            "Sayeed Ahmad works with Python Data Science and Deep Learning.",
            "Artificial Intelligence is changing the world.",
            "Machine Learning helps computers learn from data.",
            "Deep Learning uses neural networks.",
            "Generative AI creates text images and code.",
            "Sayeed Ahmad develops NLP and Generative AI applications.",
            "Python is one of the most popular programming languages.",
            "Data Science combines statistics programming and machine learning."
        ]

        dataset = TextDataset(
            texts,
            tokenizer
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True
        )

        train_model(
            model,
            dataloader,
            device,
            args.epochs
        )

        model.save_pretrained(
            "saved_model"
        )

        tokenizer.save_pretrained(
            "saved_model"
        )

        print("\nModel Saved Successfully!")

    # PREDICT
    
    elif args.predict:

        generate_text(
            model,
            tokenizer,
            args.prompt,
            device
        )

    else:
        print(
            "Use --train or --predict"
        )


if __name__ == "__main__":
    main() 