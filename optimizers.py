"""
optimizer.py
Modern PyTorch Optimizer & Scheduler
"""

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup


class OptimizerManager:

    def __init__(self, model, config):

        self.model = model

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("lr", 5e-5),
            betas=(
                config.get("beta1", 0.9),
                config.get("beta2", 0.999)
            ),
            eps=config.get("epsilon", 1e-8),
            weight_decay=config.get(
                "weight_decay",
                0.01
            )
        )

        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=config.get(
                "warmup_steps",
                1000
            ),
            num_training_steps=config.get(
                "train_steps",
                10000
            )
        )

        self.max_grad_norm = config.get(
            "gradient_clipping",
            1.0
        )

    def train_step(self, batch):

        self.model.train()

        outputs = self.model(
            input_ids=batch,
            labels=batch
        )

        loss = outputs.loss

        self.optimizer.zero_grad()

        loss.backward()

        clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )

        self.optimizer.step()

        self.scheduler.step()

        return loss.item()

    def get_lr(self):

        return self.scheduler.get_last_lr()[0]


if __name__ == "__main__":

    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(
        "gpt2"
    )

    config = {

        "lr": 5e-5,

        "beta1": 0.9,

        "beta2": 0.999,

        "epsilon": 1e-8,

        "weight_decay": 0.01,

        "warmup_steps": 100,

        "train_steps": 1000,

        "gradient_clipping": 1.0
    }

    trainer = OptimizerManager(
        model,
        config
    )

    dummy_batch = torch.randint(
        0,
        50257,
        (2, 128)
    )

    loss = trainer.train_step(
        dummy_batch
    )

    print(f"Loss: {loss:.4f}")

    print(
        f"Learning Rate: "
        f"{trainer.get_lr():.8f}"
    ) 