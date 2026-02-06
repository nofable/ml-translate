import pytest
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from ml_translate.data import MAX_LENGTH
from ml_translate.model import EncoderRNN, DecoderRNN
from ml_translate.train import train_epoch, validate_epoch, train, TrainResult, EarlyStopping


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

    @pytest.mark.slow
    def test_train_early_stopping_records_best_loss(self, training_setup, small_dataloader):
        """Test that early stopping records best validation loss."""
        encoder, decoder, _, _, _ = training_setup

        result = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=4,
            print_every=1,
            plot_every=1,
            val_dataloader=small_dataloader,
            early_stopping_patience=10,  # High patience so it won't trigger
        )

        # Should record best validation loss even if not stopped early
        assert result.best_val_loss is not None
        assert result.best_val_loss > 0

    @pytest.mark.slow
    def test_train_early_stopping_disabled(self, training_setup, small_dataloader):
        """Test that training runs full epochs without early stopping."""
        encoder, decoder, _, _, _ = training_setup

        result = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=4,
            print_every=1,
            plot_every=1,
            val_dataloader=small_dataloader,
            early_stopping_patience=None,
        )

        assert len(result.train_losses) == 4
        assert result.stopped_early is False


class TestEarlyStopping:
    def test_early_stopping_init(self):
        """Test EarlyStopping initialization."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.counter == 0
        assert es.best_loss is None

    def test_early_stopping_first_call(self):
        """Test that first call sets best_loss and returns False."""
        es = EarlyStopping(patience=3)
        result = es(1.0)
        assert result is False
        assert es.best_loss == 1.0
        assert es.counter == 0

    def test_early_stopping_improvement(self):
        """Test that improvement resets counter."""
        es = EarlyStopping(patience=3)
        es(1.0)
        es(0.9)  # Improvement
        assert es.best_loss == 0.9
        assert es.counter == 0

    def test_early_stopping_no_improvement(self):
        """Test that no improvement increments counter."""
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)  # No improvement
        assert es.best_loss == 1.0
        assert es.counter == 1

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers after patience epochs."""
        es = EarlyStopping(patience=3)
        es(1.0)
        assert es(1.1) is False  # counter=1
        assert es(1.2) is False  # counter=2
        assert es(1.3) is True   # counter=3, triggers

    def test_early_stopping_min_delta(self):
        """Test that min_delta controls improvement threshold."""
        es = EarlyStopping(patience=3, min_delta=0.1)
        es(1.0)
        # 0.95 is not enough improvement (< min_delta)
        es(0.95)
        assert es.counter == 1
        # 0.85 is enough improvement (> min_delta from 1.0)
        es(0.85)
        assert es.counter == 0
        assert es.best_loss == 0.85

    def test_early_stopping_reset_on_improvement(self):
        """Test that counter resets when improvement occurs."""
        es = EarlyStopping(patience=5)
        es(1.0)
        es(1.1)  # counter=1
        es(1.2)  # counter=2
        es(0.8)  # Improvement, counter=0
        assert es.counter == 0
        assert es.best_loss == 0.8


class TestGradientClipping:
    def test_train_epoch_with_gradient_clipping(self, training_setup, small_dataloader):
        """Test that train_epoch works with gradient clipping."""
        encoder, decoder, enc_opt, dec_opt, criterion = training_setup

        loss = train_epoch(
            small_dataloader, encoder, decoder, enc_opt, dec_opt, criterion,
            max_grad_norm=1.0,
        )

        assert isinstance(loss, float)
        assert loss > 0

    def test_train_epoch_without_gradient_clipping(self, training_setup, small_dataloader):
        """Test that train_epoch works without gradient clipping."""
        encoder, decoder, enc_opt, dec_opt, criterion = training_setup

        loss = train_epoch(
            small_dataloader, encoder, decoder, enc_opt, dec_opt, criterion,
            max_grad_norm=None,
        )

        assert isinstance(loss, float)
        assert loss > 0

    @pytest.mark.slow
    def test_train_with_gradient_clipping(self, training_setup, small_dataloader):
        """Test that train works with gradient clipping enabled."""
        encoder, decoder, _, _, _ = training_setup

        result = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=2,
            print_every=1,
            plot_every=1,
            max_grad_norm=1.0,
        )

        assert len(result.train_losses) == 2


class TestLearningRateScheduler:
    @pytest.mark.slow
    def test_train_with_scheduler(self, training_setup, small_dataloader):
        """Test that training works with learning rate scheduler."""
        encoder, decoder, _, _, _ = training_setup

        result = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=4,
            print_every=1,
            plot_every=1,
            val_dataloader=small_dataloader,
            scheduler_patience=2,
            scheduler_factor=0.5,
        )

        assert isinstance(result, TrainResult)
        assert len(result.train_losses) == 4

    @pytest.mark.slow
    def test_train_scheduler_without_validation(self, training_setup, small_dataloader):
        """Test that scheduler is ignored without validation dataloader."""
        encoder, decoder, _, _, _ = training_setup

        # Should not raise error even though scheduler_patience is set
        result = train(
            small_dataloader,
            encoder,
            decoder,
            n_epochs=2,
            print_every=1,
            plot_every=1,
            val_dataloader=None,
            scheduler_patience=2,
        )

        assert len(result.train_losses) == 2
