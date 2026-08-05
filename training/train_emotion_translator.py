"""
Train Emotion Translator with Auxiliary Latent Space Loss.

This script trains an LLM-based emotion translator using LoRA fine-tuning
with a composite loss: standard cross-entropy for text generation plus an
MSE loss that aligns the translator's hidden representations with a
pre-trained emotion bottleneck teacher's latent space.

The auxiliary loss encourages the translator to produce outputs whose
emotional content matches the target emotion in a structured 100-dimensional
latent space, rather than relying solely on surface-level token prediction.

See Section 3.2 of the paper for details on the training objective.

Usage:
    python train_emotion_translator.py --latent_weight 0.02 --output_dir ./outputs/emo100_002

Dependencies:
    torch, transformers, peft, pyyaml, tqdm
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from pathlib import Path
import json
from typing import List


# ---------------------------------------------------------------------------
# Bottleneck Head (See Section 3.1)
# ---------------------------------------------------------------------------

class BottleneckHead(nn.Module):
    """Bottleneck classifier: input_dim -> latent_dim -> num_classes.

    Architecture: Linear -> ReLU -> Linear
    The latent layer is forced to encode emotion-relevant features because
    it is the only pathway to the classification output.
    """

    def __init__(self, input_dim: int = 768, latent_dim: int = 100, num_classes: int = 7):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """Returns (latent, logits)."""
        latent = F.relu(self.encoder(x))
        logits = self.classifier(latent)
        return latent, logits

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Returns only the latent representation."""
        return F.relu(self.encoder(x))


# ---------------------------------------------------------------------------
# Fine-Tuned Emotion Teacher (See Section 3.1)
# ---------------------------------------------------------------------------

class FineTunedTeacher(nn.Module):
    """Fine-tuned teacher: adapted backbone + bottleneck head.

    The teacher provides target latent vectors for the auxiliary MSE loss.
    Both backbone and head are frozen during translator training.
    """

    def __init__(self, backbone_path, head_path, device='cpu'):
        super().__init__()
        self.base = AutoModel.from_pretrained(backbone_path)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_path)
        self.head = BottleneckHead(input_dim=768, latent_dim=100, num_classes=7)
        self.head.load_state_dict(torch.load(head_path, map_location='cpu'))
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_latent(self, texts: List[str]) -> torch.Tensor:
        """Compute 100-dim emotion latent vectors for a batch of texts."""
        actual_device = next(self.base.parameters()).device
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt",
        ).to(actual_device)
        cls_h = self.base(**inputs).last_hidden_state[:, 0, :]
        latent = self.head.get_latent(cls_h)
        return latent


# ---------------------------------------------------------------------------
# Emotion Translator Model (See Section 3.2)
# ---------------------------------------------------------------------------

class EmotionTranslator(nn.Module):
    """Emotion translator with auxiliary latent space loss.

    Loss = ce_weight * CE_loss + latent_weight * MSE_loss

    The CE loss trains standard sequence-to-sequence translation.
    The MSE loss aligns the translator's output representation with the
    teacher's emotion latent space, providing a structured training signal.
    """

    def __init__(self, base_model_name, lora_r=16, lora_alpha=32,
                 lora_dropout=0.05, target_modules=None,
                 ce_weight=1.0, latent_weight=0.02):
        super().__init__()
        self.ce_weight = ce_weight
        self.latent_weight = latent_weight

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        # Load base LLM with LoRA adapters
        llm = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(llm, lora_config)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id

        # Projection from LLM hidden size to teacher latent dim
        hidden_size = self.llm.config.hidden_size
        self.student_proj = nn.Linear(hidden_size, 100).to(torch.bfloat16)

    def forward(self, input_ids, attention_mask, labels, target_texts=None, teacher=None):
        """Compute composite loss: CE + latent MSE.

        Args:
            input_ids: Tokenized input sequence.
            attention_mask: Attention mask.
            labels: Target token IDs (-100 for prompt tokens).
            target_texts: Raw target text strings for teacher encoding.
            teacher: FineTunedTeacher instance for latent targets.

        Returns:
            dict with 'loss', 'ce_loss', 'latent_loss'.
        """
        outputs = self.llm(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            output_hidden_states=True, return_dict=True,
        )
        ce_loss = outputs.loss

        # If no teacher or weight is zero, skip latent loss
        if target_texts is None or teacher is None or self.latent_weight == 0:
            return {"loss": ce_loss, "ce_loss": ce_loss.detach(),
                    "latent_loss": torch.tensor(0.0, device=ce_loss.device)}

        # Pool hidden states over target tokens only (exclude prompt)
        hidden_states = outputs.hidden_states[-1]
        target_mask = (labels != -100) & (attention_mask == 1)
        mask = target_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        # Project student hidden states to teacher latent dim
        student_latent = self.student_proj(pooled)

        # Get teacher latent targets (no gradient)
        with torch.no_grad():
            teacher_latent = teacher.get_latent(target_texts)
        teacher_latent = teacher_latent.to(device=student_latent.device, dtype=student_latent.dtype)

        # Composite loss (See Equation 1 in Section 3.2)
        latent_loss = F.mse_loss(student_latent, teacher_latent)
        total_loss = self.ce_weight * ce_loss + self.latent_weight * latent_loss

        return {"loss": total_loss, "ce_loss": ce_loss.detach(), "latent_loss": latent_loss.detach()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """Dataset for emotion translation training.

    Expected JSONL format per line:
        {"instruction": "Translate this text from neutral to joy.",
         "input": "The store has 5 apples.",
         "output": "The wonderful store has 5 amazing apples!"}
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, 'r') as f:
            self.examples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = f"{example['instruction']}\n\nInput: {example['input']}\n\nOutput:"
        full_text = f"{prompt} {example['output']}"

        encoding = self.tokenizer(full_text, max_length=self.max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
        prompt_encoding = self.tokenizer(prompt, max_length=self.max_length,
                                         truncation=True, return_tensors='pt')
        prompt_length = prompt_encoding['input_ids'].shape[1]

        # Mask prompt tokens in labels so loss is computed only on the output
        labels = encoding['input_ids'].clone()
        labels[0, :prompt_length] = -100

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'target_text_str': example['output'],
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'target_text_str': [b['target_text_str'] for b in batch],
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train emotion translator with latent space loss")
    parser.add_argument('--latent_weight', type=float, required=True,
                        help='Weight for the auxiliary MSE loss (e.g., 0.02)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=None, help='Override num_epochs from config')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Training data directory')
    parser.add_argument('--config', type=str, default='./configs/training_config.yaml',
                        help='Path to training config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = f'cuda:{args.gpu}'

    # Load frozen teacher (provides latent targets)
    teacher = FineTunedTeacher(
        backbone_path='./models/finetuned_teacher/backbone',
        head_path='./models/finetuned_teacher/bottleneck_head.pt',
    )
    teacher = teacher.to(device)

    # Initialize translator model
    model = EmotionTranslator(
        base_model_name=config['model']['base_model'],
        lora_r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora'].get('target_modules', ["q_proj", "v_proj"]),
        ce_weight=1.0,
        latent_weight=args.latent_weight,
    )
    model = model.to(device)
    model.llm.print_trainable_parameters()

    # Datasets
    train_dataset = TranslationDataset(
        f'{args.data_dir}/train.jsonl', model.tokenizer,
        max_length=config['training']['max_length'])
    val_dataset = TranslationDataset(
        f'{args.data_dir}/val.jsonl', model.tokenizer,
        max_length=config['training']['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, collate_fn=collate_fn)

    # Optimizer and scheduler
    num_epochs = args.epochs if args.epochs else config['training'].get('num_epochs', 2)
    lr = float(config['training']['learning_rate'])
    grad_accum = config['training']['gradient_accumulation_steps']

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs // grad_accum
    warmup_steps = 100

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        teacher.eval()
        train_ce = train_lat = train_total = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            target_texts = batch['target_text_str']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          labels=labels, target_texts=target_texts, teacher=teacher)
            loss = outputs['loss'] / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            train_ce += outputs['ce_loss'].item()
            train_lat += outputs['latent_loss'].item()
            train_total += outputs['loss'].item()

            pbar.set_postfix({
                'ce': f'{outputs["ce_loss"].item():.4f}',
                'lat': f'{outputs["latent_loss"].item():.4f}',
                'total': f'{outputs["loss"].item():.4f}',
            })

        avg_train_ce = train_ce / len(train_loader)
        avg_train_lat = train_lat / len(train_loader)

        # Validation
        model.eval()
        val_ce = val_lat = val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                target_texts = batch['target_text_str']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                              labels=labels, target_texts=target_texts, teacher=teacher)
                val_ce += outputs['ce_loss'].item()
                val_lat += outputs['latent_loss'].item()
                val_total += outputs['loss'].item()

        avg_val_ce = val_ce / len(val_loader)
        avg_val_lat = val_lat / len(val_loader)
        avg_val_total = val_total / len(val_loader)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - CE: {avg_train_ce:.6f}, Latent: {avg_train_lat:.6f}")
        print(f"  Val   - CE: {avg_val_ce:.6f}, Latent: {avg_val_lat:.6f}, Total: {avg_val_total:.6f}")

        # Save checkpoint per epoch
        epoch_dir = Path(args.output_dir) / f"epoch{epoch+1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.llm.save_pretrained(epoch_dir)
        model.tokenizer.save_pretrained(epoch_dir)
        torch.save(model.student_proj.state_dict(), epoch_dir / "student_projection.pt")

        # Track best model
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            best_dir = Path(args.output_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.llm.save_pretrained(best_dir)
            model.tokenizer.save_pretrained(best_dir)
            torch.save(model.student_proj.state_dict(), best_dir / "student_projection.pt")
            print(f"  New best model (val_loss: {avg_val_total:.6f})")

    print(f"\nTraining complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
