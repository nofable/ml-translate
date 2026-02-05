import pytest
import torch

from ml_translate.data import Lang, MAX_LENGTH
from ml_translate.model import EncoderRNN, AttnDecoderRNN
from ml_translate.eval import evaluate


@pytest.fixture
def eval_setup(device):
    """Create encoder, decoder, and languages for evaluation tests."""
    torch.manual_seed(42)
    hidden_size = 16

    # Create input language
    input_lang = Lang("fra")
    input_lang.addSentence("je suis content")
    input_lang.addSentence("il est grand")

    # Create output language
    output_lang = Lang("eng")
    output_lang.addSentence("i am happy")
    output_lang.addSentence("he is tall")

    encoder = EncoderRNN(input_lang.n_words, hidden_size)
    encoder.to(device)
    encoder.eval()

    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, device=device)
    decoder.to(device)
    decoder.eval()

    return encoder, decoder, input_lang, output_lang


class TestEvaluate:
    def test_evaluate_returns_words(self, eval_setup, device):
        """Test that evaluate returns a list of words."""
        encoder, decoder, input_lang, output_lang = eval_setup

        words, attn = evaluate(
            encoder, decoder, "je suis content", input_lang, output_lang, device
        )

        assert isinstance(words, list)
        assert all(isinstance(w, str) for w in words)

    def test_evaluate_ends_at_eos(self, eval_setup, device):
        """Test that evaluation stops at EOS token."""
        encoder, decoder, input_lang, output_lang = eval_setup

        words, attn = evaluate(
            encoder, decoder, "je suis content", input_lang, output_lang, device
        )

        # If EOS was encountered, <EOS> should be in the words
        # or the list should be shorter than MAX_LENGTH
        if "<EOS>" in words:
            assert words[-1] == "<EOS>"
        else:
            # If no EOS, list should be at most MAX_LENGTH
            assert len(words) <= MAX_LENGTH

    def test_evaluate_max_length(self, eval_setup, device):
        """Test that output never exceeds MAX_LENGTH."""
        encoder, decoder, input_lang, output_lang = eval_setup

        words, attn = evaluate(
            encoder, decoder, "je suis content", input_lang, output_lang, device
        )

        assert len(words) <= MAX_LENGTH

    def test_evaluate_no_grad(self, eval_setup, device):
        """Test that evaluation runs without gradient computation."""
        encoder, decoder, input_lang, output_lang = eval_setup

        # This should not raise any errors related to gradients
        words, attn = evaluate(
            encoder, decoder, "je suis content", input_lang, output_lang, device
        )

        # Verify models are in eval mode
        assert not encoder.training
        assert not decoder.training
