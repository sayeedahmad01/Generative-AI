"""
trainer.py
Modern GPT Trainer
PyTorch + Transformers
"""

import math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)


class GPTTrainer:

    def __init__(self, config):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"]
        )

        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name"]
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            betas=(0.9, 0.999),
            weight_decay=config["weight_decay"]
        )

        self.scheduler = (
            get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config["warmup_steps"],
                num_training_steps=config["train_steps"]
            )
        )

        self.max_grad_norm = config.get(
            "gradient_clipping",
            1.0
        )

    def train_step(self, batch):

        self.model.train()

        batch = batch.to(self.device)

        outputs = self.model(
            input_ids=batch,
            labels=batch
        )

        loss = outputs.loss

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )

        self.optimizer.step()

        self.scheduler.step()

        return loss.item()

    def evaluate(self, batch):

        self.model.eval()

        with torch.no_grad():

            batch = batch.to(self.device)

            outputs = self.model(
                input_ids=batch,
                labels=batch
            )

            loss = outputs.loss

            perplexity = torch.exp(loss)

        return {
            "loss": loss.item(),
            "perplexity": perplexity.item()
        }

    def generate(
        self,
        prompt,
        max_tokens=100
    ):

        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

    def save(self, path):

        self.model.save_pretrained(path)

        self.tokenizer.save_pretrained(path)

    def load(self, path):

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(path)
            .to(self.device)
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(path)
        )


if __name__ == "__main__":

    config = {

        "model_name": "gpt2",

        "lr": 5e-5,

        "weight_decay": 0.01,

        "warmup_steps": 100,

        "train_steps": 1000,

        "gradient_clipping": 1.0
    }

    trainer = GPTTrainer(config)

    dummy_batch = torch.randint(
        0,
        50257,
        (2, 128)
    )

    loss = trainer.train_step(
        dummy_batch
    )

    print(f"Training Loss: {loss:.4f}")

    metrics = trainer.evaluate(
        dummy_batch
    )

    print(metrics)

    text = trainer.generate(
        "Artificial Intelligence is"
    )

    print("\nGenerated Text:\n")
    print(text) 