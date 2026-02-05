import logging
import time
from dataclasses import dataclass, field

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ml_translate.config import default_config
from ml_translate.model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from ml_translate.utils import timeSince

logger = logging.getLogger(__name__)


def train_epoch(
    dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: DecoderRNN | AttnDecoderRNN,
    encoder_optimizer: Optimizer,
    decoder_optimizer: Optimizer,
    criterion: nn.Module,
) -> float:
    """Train for one epoch.

    Args:
        dataloader: Training data loader.
        encoder: Encoder model.
        decoder: Decoder model.
        encoder_optimizer: Optimizer for encoder.
        decoder_optimizer: Optimizer for decoder.
        criterion: Loss function.

    Returns:
        Average loss over the epoch.
    """
    encoder.train()
    decoder.train()

    total_loss = 0.0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(
    dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: DecoderRNN | AttnDecoderRNN,
    criterion: nn.Module,
) -> float:
    """Validate for one epoch without computing gradients.

    Args:
        dataloader: Validation data loader.
        encoder: Encoder model.
        decoder: Decoder model.
        criterion: Loss function.

    Returns:
        Average validation loss over the epoch.
    """
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


@dataclass
class TrainResult:
    """Results from training, including loss history."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def train(
    train_dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: DecoderRNN | AttnDecoderRNN,
    n_epochs: int,
    learning_rate: float = default_config.learning_rate,
    print_every: int = 100,
    plot_every: int = 100,
    val_dataloader: DataLoader | None = None,
) -> TrainResult:
    """Train the encoder-decoder model.

    Args:
        train_dataloader: Training data loader.
        encoder: Encoder model.
        decoder: Decoder model.
        n_epochs: Number of training epochs.
        learning_rate: Learning rate for Adam optimizer.
        print_every: Print progress every N epochs.
        plot_every: Record loss every N epochs.
        val_dataloader: Optional validation data loader. If provided,
            validation loss is computed after each epoch.

    Returns:
        TrainResult containing train and validation loss histories.
    """
    start = time.time()
    result = TrainResult()
    print_loss_total = 0.0  # Reset every print_every
    plot_loss_total = 0.0  # Reset every plot_every
    print_val_loss_total = 0.0
    plot_val_loss_total = 0.0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += train_loss
        plot_loss_total += train_loss

        # Compute validation loss if validation dataloader provided
        val_loss = None
        if val_dataloader is not None:
            val_loss = validate_epoch(val_dataloader, encoder, decoder, criterion)
            print_val_loss_total += val_loss
            plot_val_loss_total += val_loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0.0

            if val_loss is not None:
                print_val_loss_avg = print_val_loss_total / print_every
                print_val_loss_total = 0.0
                logger.info(
                    "%s (%d %.0f%%) train=%.4f val=%.4f",
                    timeSince(start, epoch / n_epochs),
                    epoch,
                    epoch / n_epochs * 100,
                    print_loss_avg,
                    print_val_loss_avg,
                )
            else:
                logger.info(
                    "%s (%d %.0f%%) %.4f",
                    timeSince(start, epoch / n_epochs),
                    epoch,
                    epoch / n_epochs * 100,
                    print_loss_avg,
                )

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            result.train_losses.append(plot_loss_avg)
            plot_loss_total = 0.0

            if val_dataloader is not None:
                plot_val_loss_avg = plot_val_loss_total / plot_every
                result.val_losses.append(plot_val_loss_avg)
                plot_val_loss_total = 0.0

    return result
