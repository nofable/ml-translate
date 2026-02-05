import pytest
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from ml_translate.data import MAX_LENGTH
from ml_translate.model import EncoderRNN, DecoderRNN
from ml_translate.train import train_epoch, train


@pytest.fixture
def training_setup(sample_lang, small_hidden_size, device):
    """Create encoder, decoder, optimizers, and criterion for training tests."""
    torch.manual_seed(42)

    encoder = EncoderRNN(sample_lang.n_words, small_hidden_size)
    encoder.to(device)
    decoder = DecoderRNN(small_hidden_size, sample_lang.n_words, device=device)
    decoder.to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    return encoder, decoder, encoder_optimizer, decoder_optimizer, criterion


@pytest.fixture
def small_dataloader(sample_lang, device):
    """Create a small dataloader for testing."""
    torch.manual_seed(42)
    n_samples = 4
    input_ids = torch.randint(
        0, sample_lang.n_words, (n_samples, MAX_LENGTH), device=device
    )
    target_ids = torch.randint(
        0, sample_lang.n_words, (n_samples, MAX_LENGTH), device=device
    )

    dataset = TensorDataset(input_ids, target_ids)
    dataloader = DataLoader(dataset, batch_size=2)
    return dataloader


class TestTrainEpoch:
    def test_train_epoch_returns_loss(self, training_setup, small_dataloader):
        """Test that train_epoch returns a loss value."""
        encoder, decoder, enc_opt, dec_opt, criterion = training_setup

        loss = train_epoch(
            small_dataloader, encoder, decoder, enc_opt, dec_opt, criterion
        )

        assert isinstance(loss, float)

    def test_train_epoch_loss_positive(self, training_setup, small_dataloader):
        """Test that loss is positive."""
        encoder, decoder, enc_opt, dec_opt, criterion = training_setup

        loss = train_epoch(
            small_dataloader, encoder, decoder, enc_opt, dec_opt, criterion
        )

        assert loss > 0


class TestTrain:
    @pytest.mark.slow
    def test_train_returns_plot_losses(self, training_setup, small_dataloader):
        """Test that train returns a list of losses."""
        encoder, decoder, _, _, _ = training_setup

        plot_losses = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=2,
            print_every=1,
            plot_every=1,
        )

        assert isinstance(plot_losses, list)
        assert len(plot_losses) > 0

    @pytest.mark.slow
    def test_train_plot_losses_length(self, training_setup, small_dataloader):
        """Test plot_losses length matches epochs/plot_every."""
        encoder, decoder, _, _, _ = training_setup

        n_epochs = 4
        plot_every = 2

        plot_losses = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=n_epochs,
            print_every=1,
            plot_every=plot_every,
        )

        expected_length = n_epochs // plot_every
        assert len(plot_losses) == expected_length
