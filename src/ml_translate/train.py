import time

from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ml_translate.model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from ml_translate.utils import timeSince


def train_epoch(
    dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: DecoderRNN | AttnDecoderRNN,
    encoder_optimizer: Optimizer,
    decoder_optimizer: Optimizer,
    criterion: nn.Module,
) -> float:
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


def train(
    train_dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: DecoderRNN | AttnDecoderRNN,
    n_epochs: int,
    learning_rate: float = 0.001,
    print_every: int = 100,
    plot_every: int = 100,
) -> list[float]:
    start = time.time()
    plot_losses: list[float] = []
    print_loss_total = 0.0  # Reset every print_every
    plot_loss_total = 0.0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0.0
            print(
                f"{timeSince(start, epoch / n_epochs)} ({epoch} {epoch / n_epochs * 100}) {print_loss_avg}"
            )

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0.0

    return plot_losses
