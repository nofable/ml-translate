import pytest
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from ml_translate.data import MAX_LENGTH
from ml_translate.model import EncoderRNN, DecoderRNN
from ml_translate.train import train_epoch, validate_epoch, train, TrainResult


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

    def test_train_epoch_sets_train_mode(self, training_setup, small_dataloader):
        """Test that train_epoch sets models to train mode."""
        encoder, decoder, enc_opt, dec_opt, criterion = training_setup

        # Set to eval mode first
        encoder.eval()
        decoder.eval()

        train_epoch(small_dataloader, encoder, decoder, enc_opt, dec_opt, criterion)

        assert encoder.training is True
        assert decoder.training is True


class TestValidateEpoch:
    def test_validate_epoch_returns_loss(self, training_setup, small_dataloader):
        """Test that validate_epoch returns a loss value."""
        encoder, decoder, _, _, criterion = training_setup

        loss = validate_epoch(small_dataloader, encoder, decoder, criterion)

        assert isinstance(loss, float)

    def test_validate_epoch_loss_positive(self, training_setup, small_dataloader):
        """Test that validation loss is positive."""
        encoder, decoder, _, _, criterion = training_setup

        loss = validate_epoch(small_dataloader, encoder, decoder, criterion)

        assert loss > 0

    def test_validate_epoch_sets_eval_mode(self, training_setup, small_dataloader):
        """Test that validate_epoch sets models to eval mode."""
        encoder, decoder, _, _, criterion = training_setup

        # Set to train mode first
        encoder.train()
        decoder.train()

        validate_epoch(small_dataloader, encoder, decoder, criterion)

        assert encoder.training is False
        assert decoder.training is False

    def test_validate_epoch_no_gradient(self, training_setup, small_dataloader):
        """Test that validate_epoch doesn't compute gradients."""
        encoder, decoder, _, _, criterion = training_setup

        # Zero out any existing gradients
        encoder.zero_grad()
        decoder.zero_grad()

        validate_epoch(small_dataloader, encoder, decoder, criterion)

        # Check that no gradients were computed
        for param in encoder.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0
        for param in decoder.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0


class TestTrain:
    @pytest.mark.slow
    def test_train_returns_train_result(self, training_setup, small_dataloader):
        """Test that train returns a TrainResult."""
        encoder, decoder, _, _, _ = training_setup

        result = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=2,
            print_every=1,
            plot_every=1,
        )

        assert isinstance(result, TrainResult)
        assert isinstance(result.train_losses, list)
        assert len(result.train_losses) > 0

    @pytest.mark.slow
    def test_train_losses_length(self, training_setup, small_dataloader):
        """Test train_losses length matches epochs/plot_every."""
        encoder, decoder, _, _, _ = training_setup

        n_epochs = 4
        plot_every = 2

        result = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=n_epochs,
            print_every=1,
            plot_every=plot_every,
        )

        expected_length = n_epochs // plot_every
        assert len(result.train_losses) == expected_length

    @pytest.mark.slow
    def test_train_without_validation(self, training_setup, small_dataloader):
        """Test that train without validation has empty val_losses."""
        encoder, decoder, _, _, _ = training_setup

        result = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=2,
            print_every=1,
            plot_every=1,
        )

        assert len(result.val_losses) == 0

    @pytest.mark.slow
    def test_train_with_validation(self, training_setup, small_dataloader):
        """Test that train with validation populates val_losses."""
        encoder, decoder, _, _, _ = training_setup

        # Use same dataloader for train and val in this test
        result = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=2,
            print_every=1,
            plot_every=1,
            val_dataloader=small_dataloader,
        )

        assert len(result.val_losses) == 2
        assert all(isinstance(loss, float) for loss in result.val_losses)
        assert all(loss > 0 for loss in result.val_losses)

    @pytest.mark.slow
    def test_train_val_losses_length_matches_train(self, training_setup, small_dataloader):
        """Test that val_losses length matches train_losses."""
        encoder, decoder, _, _, _ = training_setup

        n_epochs = 4
        plot_every = 2

        result = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=n_epochs,
            print_every=1,
            plot_every=plot_every,
            val_dataloader=small_dataloader,
        )

        assert len(result.val_losses) == len(result.train_losses)
