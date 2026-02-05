import pytest
import torch

from ml_translate.data import Lang
from ml_translate.model import EncoderRNN, DecoderRNN, AttnDecoderRNN


@pytest.fixture
def device():
    """CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_lang():
    """Pre-populated Lang object for testing."""
    lang = Lang("test")
    lang.addSentence("hello world")
    lang.addSentence("this is a test")
    lang.addSentence("hello again")
    return lang


@pytest.fixture
def sample_pairs():
    """Test sentence pairs."""
    return [
        ["je suis content", "i am happy"],
        ["il est grand", "he is tall"],
        ["elle est belle", "she is beautiful"],
    ]


@pytest.fixture
def small_hidden_size():
    """Small hidden size for fast tests."""
    return 16


@pytest.fixture
def small_batch_size():
    """Small batch size for fast tests."""
    return 2


@pytest.fixture
def encoder(sample_lang, small_hidden_size, device):
    """Small encoder instance for testing."""
    torch.manual_seed(42)
    enc = EncoderRNN(sample_lang.n_words, small_hidden_size)
    enc.to(device)
    enc.eval()
    return enc


@pytest.fixture
def decoder(sample_lang, small_hidden_size, device):
    """Small decoder instance for testing."""
    torch.manual_seed(42)
    dec = DecoderRNN(small_hidden_size, sample_lang.n_words, device=device)
    dec.to(device)
    dec.eval()
    return dec


@pytest.fixture
def attn_decoder(sample_lang, small_hidden_size, device):
    """Small attention decoder instance for testing."""
    torch.manual_seed(42)
    dec = AttnDecoderRNN(small_hidden_size, sample_lang.n_words, device=device)
    dec.to(device)
    dec.eval()
    return dec
