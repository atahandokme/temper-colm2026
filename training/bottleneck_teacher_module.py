"""
Bottleneck Teacher: 100-dim Emotion Latent Space
================================================

Adds a bottleneck layer to j-hartmann to get rich emotion embeddings.

Architecture:
    [CLS] (768-dim) → Linear (768→100) → ReLU → Linear (100→7) → Softmax

The 100-dim bottleneck is forced to encode emotion-relevant features
because it's the only path to the classifier.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from transformers import AutoModel, AutoTokenizer


class BottleneckHead(nn.Module):
    """Bottleneck classifier: 768 → 100 → 7"""

    def __init__(self, input_dim: int = 768, latent_dim: int = 100, num_classes: int = 7):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (latent, logits)"""
        latent = F.relu(self.encoder(x))
        logits = self.classifier(latent)
        return latent, logits

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Returns only the 100-dim latent"""
        return F.relu(self.encoder(x))


class BottleneckTeacher(nn.Module):
    """
    j-hartmann base + bottleneck head for 100-dim emotion latents.

    Must be trained before use! See scripts/02_train_bottleneck.py
    """

    def __init__(
        self,
        latent_dim: int = 100,
        device: torch.device = None,
        checkpoint_path: str = None,
    ):
        super().__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim

        # Load j-hartmann base (frozen)
        print("Loading j-hartmann base model...")
        self.base = AutoModel.from_pretrained(
            "j-hartmann/emotion-english-distilroberta-base", use_safetensors=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "j-hartmann/emotion-english-distilroberta-base"
        )

        # Freeze base
        for param in self.base.parameters():
            param.requires_grad = False

        # Bottleneck head (trainable)
        self.head = BottleneckHead(
            input_dim=self.base.config.hidden_size,  # 768
            latent_dim=latent_dim,
            num_classes=7,
        )

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading bottleneck head from {checkpoint_path}")
            self.head.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

        self.to(self.device)
        self.eval()

        # Emotion labels
        self.labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    @torch.no_grad()
    def get_latent(self, texts: List[str]) -> torch.Tensor:
        """Get 100-dim emotion latent for texts."""
        # Use actual device of model parameters (handles multi-GPU correctly)
        actual_device = next(self.base.parameters()).device

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(actual_device)

        # Get [CLS] from base
        outputs = self.base(**inputs)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

        # Get latent from bottleneck
        latent = self.head.get_latent(cls_hidden)  # [batch, 100]
        return latent

    @torch.no_grad()
    def get_probs(self, texts: List[str], temperature: float = 1.0) -> torch.Tensor:
        """Get emotion probabilities."""
        # Use actual device of model parameters
        actual_device = next(self.base.parameters()).device

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(actual_device)

        outputs = self.base(**inputs)
        cls_hidden = outputs.last_hidden_state[:, 0, :]

        _, logits = self.head(cls_hidden)
        probs = F.softmax(logits / temperature, dim=-1)
        return probs

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict emotion labels."""
        probs = self.get_probs(texts)
        predictions = []
        for p in probs:
            idx = p.argmax().item()
            predictions.append((self.labels[idx], p[idx].item()))
        return predictions
