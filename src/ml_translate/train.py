import logging
import time
from dataclasses import dataclass, field

import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    max_grad_norm: float | None = None,
) -> float:
    """Train for one epoch.

    Args:
        dataloader: Training data loader.
        encoder: Encoder model.
        decoder: Decoder model.
        encoder_optimizer: Optimizer for encoder.
        decoder_optimizer: Optimizer for decoder.
        criterion: Loss function.
        max_grad_norm: If set, clip gradients to this max norm.

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

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(decoder.parameters(), max_grad_norm)

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
    stopped_early: bool = False
    best_val_loss: float | None = None


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def train(
    train_dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: DecoderRNN | AttnDecoderRNN,
    n_epochs: int,
    learning_rate: float = default_config.learning_rate,
    print_every: int = 100,
    plot_every: int = 100,
    val_dataloader: DataLoader | None = None,
    early_stopping_patience: int | None = None,
    scheduler_patience: int | None = None,
    scheduler_factor: float = 0.5,
    max_grad_norm: float | None = None,
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
        early_stopping_patience: If set, stop training after this many epochs
            without validation loss improvement. Requires val_dataloader.
        scheduler_patience: If set, reduce learning rate after this many epochs
            without validation loss improvement. Requires val_dataloader.
        scheduler_factor: Factor to reduce learning rate by (default 0.5).
        max_grad_norm: If set, clip gradients to this max norm.
            Common values are 1.0 or 5.0. Default is None (no clipping).

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

    early_stopping = None
    if early_stopping_patience is not None and val_dataloader is not None:
        early_stopping = EarlyStopping(patience=early_stopping_patience)

    # Learning rate schedulers
    encoder_scheduler = None
    decoder_scheduler = None
    if scheduler_patience is not None and val_dataloader is not None:
        encoder_scheduler = ReduceLROnPlateau(
            encoder_optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
        )
        decoder_scheduler = ReduceLROnPlateau(
            decoder_optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
        )

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
            max_grad_norm,
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
                progress = epoch / n_epochs * 100
                logger.info(
                    f"{timeSince(start, epoch / n_epochs)} "
                    f"(epoch {epoch}, {progress:.0f}%) "
                    f"train_loss={print_loss_avg:.3f} val_loss={print_val_loss_avg:.3f}"
                )
            else:
                progress = epoch / n_epochs * 100
                logger.info(
                    f"{timeSince(start, epoch / n_epochs)} "
                    f"(epoch {epoch}, {progress:.0f}%) "
                    f"train_loss={print_loss_avg:.3f}"
                )

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            result.train_losses.append(plot_loss_avg)
            plot_loss_total = 0.0

            if val_dataloader is not None:
                plot_val_loss_avg = plot_val_loss_total / plot_every
                result.val_losses.append(plot_val_loss_avg)
                plot_val_loss_total = 0.0

        # Step learning rate schedulers
        if encoder_scheduler is not None and val_loss is not None:
            old_lr = encoder_optimizer.param_groups[0]["lr"]
            encoder_scheduler.step(val_loss)
            decoder_scheduler.step(val_loss)
            new_lr = encoder_optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                logger.info(f"Reducing learning rate to {new_lr:.6f}")

        # Check early stopping
        if early_stopping is not None and val_loss is not None:
            if early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                result.stopped_early = True
                break

    # Record best validation loss
    if early_stopping is not None:
        result.best_val_loss = early_stopping.best_loss

    return result
