import pytest
import torch

from ml_translate.model import (
    BahdanauAttention,
    LuongAttention,
    AttnDecoderRNN,
    EncoderRNN,
)


class TestBahdanauAttention:
    def test_bahdanau_output_shape(self):
        """Test that Bahdanau attention returns correct shapes."""
        hidden_size = 64
        batch_size = 4
        seq_len = 10

        attention = BahdanauAttention(hidden_size)
        query = torch.randn(batch_size, 1, hidden_size)
        keys = torch.randn(batch_size, seq_len, hidden_size)

        context, weights = attention(query, keys)

        assert context.shape == (batch_size, 1, hidden_size)
        assert weights.shape == (batch_size, 1, seq_len)

    def test_bahdanau_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        attention = BahdanauAttention(hidden_size=32)
        query = torch.randn(2, 1, 32)
        keys = torch.randn(2, 8, 32)

        _, weights = attention(query, keys)

        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestLuongAttention:
    @pytest.mark.parametrize("method", ["dot", "general", "concat"])
    def test_luong_output_shape(self, method):
        """Test that Luong attention returns correct shapes for all methods."""
        hidden_size = 64
        batch_size = 4
        seq_len = 10

        attention = LuongAttention(hidden_size, method=method)
        query = torch.randn(batch_size, 1, hidden_size)
        keys = torch.randn(batch_size, seq_len, hidden_size)

        context, weights = attention(query, keys)

        assert context.shape == (batch_size, 1, hidden_size)
        assert weights.shape == (batch_size, 1, seq_len)

    @pytest.mark.parametrize("method", ["dot", "general", "concat"])
    def test_luong_weights_sum_to_one(self, method):
        """Test that attention weights sum to 1 for all methods."""
        attention = LuongAttention(hidden_size=32, method=method)
        query = torch.randn(2, 1, 32)
        keys = torch.randn(2, 8, 32)

        _, weights = attention(query, keys)

        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_luong_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown attention method"):
            LuongAttention(hidden_size=32, method="invalid")


class TestAttnDecoderRNN:
    @pytest.fixture
    def decoder_params(self):
        return {
            "hidden_size": 64,
            "output_size": 100,
            "device": torch.device("cpu"),
        }

    def test_decoder_with_bahdanau(self, decoder_params):
        """Test decoder with Bahdanau attention."""
        decoder = AttnDecoderRNN(**decoder_params, attention_type="bahdanau")
        assert isinstance(decoder.attention, BahdanauAttention)

    def test_decoder_with_luong_dot(self, decoder_params):
        """Test decoder with Luong dot attention."""
        decoder = AttnDecoderRNN(**decoder_params, attention_type="luong_dot")
        assert isinstance(decoder.attention, LuongAttention)
        assert decoder.attention.method == "dot"

    def test_decoder_with_luong_general(self, decoder_params):
        """Test decoder with Luong general attention."""
        decoder = AttnDecoderRNN(**decoder_params, attention_type="luong_general")
        assert isinstance(decoder.attention, LuongAttention)
        assert decoder.attention.method == "general"

    def test_decoder_with_luong_concat(self, decoder_params):
        """Test decoder with Luong concat attention."""
        decoder = AttnDecoderRNN(**decoder_params, attention_type="luong_concat")
        assert isinstance(decoder.attention, LuongAttention)
        assert decoder.attention.method == "concat"

    def test_decoder_invalid_attention_type(self, decoder_params):
        """Test that invalid attention type raises error."""
        with pytest.raises(ValueError, match="Unknown attention type"):
            AttnDecoderRNN(**decoder_params, attention_type="invalid")

    @pytest.mark.parametrize("attention_type", ["bahdanau", "luong_dot", "luong_general", "luong_concat"])
    def test_decoder_forward_pass(self, decoder_params, attention_type):
        """Test forward pass works for all attention types."""
        batch_size = 4
        seq_len = 8

        encoder = EncoderRNN(input_size=100, hidden_size=decoder_params["hidden_size"])
        decoder = AttnDecoderRNN(**decoder_params, attention_type=attention_type)

        # Create dummy input
        input_tensor = torch.randint(0, 100, (batch_size, seq_len))
        target_tensor = torch.randint(0, 100, (batch_size, seq_len))

        # Forward pass
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, attentions = decoder(
            encoder_outputs, encoder_hidden, target_tensor
        )

        assert decoder_outputs.shape == (batch_size, seq_len, decoder_params["output_size"])
        assert attentions.shape[0] == batch_size
        assert attentions.shape[1] == seq_len
