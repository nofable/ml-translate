"""Pre-trained embedding utilities."""

import logging

import torch
from torch import nn, Tensor
from torchtext.vocab import GloVe

from ml_translate.data import Lang

logger = logging.getLogger(__name__)


def load_glove_embeddings(
    lang: Lang,
    name: str = "6B",
    dim: int = 100,
) -> Tensor:
    """Load GloVe embeddings for a vocabulary.

    Args:
        lang: Language object containing the vocabulary.
        name: GloVe variant ("6B", "42B", "840B", "twitter.27B").
        dim: Embedding dimension (50, 100, 200, 300 for 6B).

    Returns:
        Tensor of shape (vocab_size, dim) with embeddings.
    """
    logger.info(f"Loading GloVe {name} with dim={dim}")
    glove = GloVe(name=name, dim=dim)

    vocab_size = lang.n_words
    embedding_matrix = torch.zeros(vocab_size, dim)

    found = 0
    for word, idx in lang.word2index.items():
        if word in glove.stoi:
            embedding_matrix[idx] = glove.vectors[glove.stoi[word]]
            found += 1
        else:
            # Random initialization for OOV words
            embedding_matrix[idx] = torch.randn(dim) * 0.1

    # Initialize SOS and EOS tokens randomly
    embedding_matrix[0] = torch.randn(dim) * 0.1  # SOS
    embedding_matrix[1] = torch.randn(dim) * 0.1  # EOS

    coverage = found / (vocab_size - 2) * 100  # Exclude SOS/EOS
    logger.info(
        f"Found {found}/{vocab_size - 2} words in GloVe ({coverage:.1f}% coverage)"
    )

    return embedding_matrix


class PretrainedEmbedding(nn.Module):
    """Embedding layer with optional pre-trained weights."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        pretrained_weights: Tensor | None = None,
        freeze: bool = False,
    ):
        """
        Args:
            num_embeddings: Size of vocabulary.
            embedding_dim: Dimension of embeddings (must match hidden_size).
            pretrained_weights: Optional pre-trained embedding tensor.
            freeze: If True, embeddings are not updated during training.
        """
        super().__init__()

        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=freeze
            )
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)
