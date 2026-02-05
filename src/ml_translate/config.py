"""Centralized configuration for ml-translate."""

from dataclasses import dataclass


@dataclass
class TranslationConfig:
    """Configuration settings for the translation model."""

    max_length: int = 10
    dropout_p: float = 0.1
    learning_rate: float = 0.001
    hidden_size: int = 128
    batch_size: int = 32


# Default configuration instance
default_config = TranslationConfig()
