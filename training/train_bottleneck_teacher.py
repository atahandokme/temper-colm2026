"""
Train the Bottleneck Emotion Teacher.

Trains a 768 -> 100 -> 7 bottleneck head on top of a frozen pre-trained
emotion classification backbone. The 100-dimensional bottleneck layer is
forced to learn a compact emotion representation because it is the sole
pathway to the 7-class classifier output.

Once trained, the bottleneck latent space serves as the target for the
auxiliary MSE loss during emotion translator training (see Section 3.1).

Architecture:
    [CLS] embedding (768-dim) -> Linear(768, 100) -> ReLU -> Linear(100, 7)

Usage:
    python train_bottleneck_teacher.py

Dependencies:
    torch, transformers, tqdm
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from typing import Tuple


# ---------------------------------------------------------------------------
# Bottleneck Head (See Section 3.1)
# ---------------------------------------------------------------------------

class BottleneckHead(nn.Module):
    """Bottleneck classifier: input_dim -> latent_dim -> num_classes.

    The latent layer provides a structured emotion representation that
    captures more information than the 7-class label alone.
    """

    def __init__(self, input_dim: int = 768, latent_dim: int = 100, num_classes: int = 7):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (latent, logits)."""
        latent = F.relu(self.encoder(x))
        logits = self.classifier(latent)
        return latent, logits

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Returns only the latent representation."""
        return F.relu(self.encoder(x))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmotionTextDataset(Dataset):
    """Dataset for training the bottleneck head on emotion classification.

    Extracts (text, emotion_label) pairs from translation training data,
    using the target output text and the target emotion from the instruction.
    """

    EMOTION_MAP = {
        'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
        'neutral': 4, 'sadness': 5, 'surprise': 6
    }

    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            examples = [json.loads(line) for line in f]

        # Extract (text, emotion) pairs from target outputs
        self.samples = []
        for ex in examples:
            instruction = ex['instruction']
            target_text = ex['output']

            # Extract target emotion from instruction
            # Format: "Translate this text from X to Y."
            if ' to ' in instruction:
                target_emotion = instruction.split(' to ')[1].rstrip('.')
                if target_emotion in self.EMOTION_MAP:
                    self.samples.append({
                        'text': target_text,
                        'label': self.EMOTION_MAP[target_emotion],
                        'emotion': target_emotion,
                    })

        # Print distribution
        emotion_counts = {}
        for s in self.samples:
            emo = s['emotion']
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        print(f"Loaded {len(self.samples)} samples. Distribution:")
        for emo in sorted(emotion_counts.keys()):
            print(f"  {emo}: {emotion_counts[emo]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {
        'texts': [item['text'] for item in batch],
        'labels': torch.tensor([item['label'] for item in batch]),
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_bottleneck(train_loader, val_loader, base_model, tokenizer,
                     head, num_epochs, lr, device='cuda'):
    """Train the bottleneck head with frozen backbone.

    The backbone (e.g., DistilRoBERTa fine-tuned on emotion) is frozen.
    Only the bottleneck head parameters are updated.
    """
    base_model = base_model.to(device)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        head.train()
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            texts = batch['texts']
            labels = batch['labels'].to(device)

            # Get [CLS] embeddings from frozen backbone
            with torch.no_grad():
                inputs = tokenizer(
                    texts, padding=True, truncation=True,
                    max_length=256, return_tensors="pt",
                ).to(device)
                outputs = base_model(**inputs)
                cls_hidden = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

            # Forward through bottleneck head (gradients flow here)
            optimizer.zero_grad()
            latent, logits = head(cls_hidden)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc': f'{train_correct/train_total:.2%}'})

        train_acc = train_correct / train_total

        # Validation
        head.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                texts = batch['texts']
                labels = batch['labels'].to(device)
                inputs = tokenizer(
                    texts, padding=True, truncation=True,
                    max_length=256, return_tensors="pt",
                ).to(device)
                outputs = base_model(**inputs)
                cls_hidden = outputs.last_hidden_state[:, 0, :]
                _, logits = head(cls_hidden)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2%}, Val Acc = {val_acc:.2%}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = Path('./outputs/bottleneck_checkpoint')
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(head.state_dict(), save_dir / 'best_model.pt')
            print(f"  Saved best model (val_acc = {val_acc:.2%})")

    return head, best_val_acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load emotion classification backbone (frozen during training)
    base_model = AutoModel.from_pretrained(
        "j-hartmann/emotion-english-distilroberta-base"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "j-hartmann/emotion-english-distilroberta-base"
    )

    # Load datasets
    train_dataset = EmotionTextDataset('./data/train.jsonl')
    val_dataset = EmotionTextDataset('./data/val.jsonl')

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=4, collate_fn=collate_fn)

    # Create bottleneck head: 768 -> 100 -> 7
    head = BottleneckHead(
        input_dim=base_model.config.hidden_size,  # 768 for DistilRoBERTa
        latent_dim=100,
        num_classes=7,
    )

    # Train
    head, best_val_acc = train_bottleneck(
        train_loader, val_loader, base_model, tokenizer, head,
        num_epochs=5, lr=1e-3, device=device
    )

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2%}")


if __name__ == "__main__":
    main()
