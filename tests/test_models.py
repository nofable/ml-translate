import torch

from ml_translate.data import MAX_LENGTH
from ml_translate.model import EncoderRNN, DecoderRNN, BahdanauAttention, AttnDecoderRNN


class TestEncoderRNN:
    def test_encoder_output_shape(
        self, encoder, sample_lang, small_hidden_size, device
    ):
        """Test encoder output shape is (batch, seq_len, hidden_size)."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 5
        input_tensor = torch.randint(
            0, sample_lang.n_words, (batch_size, seq_len), device=device
        )

        output, hidden = encoder(input_tensor)

        assert output.shape == (batch_size, seq_len, small_hidden_size)

    def test_encoder_hidden_shape(
        self, encoder, sample_lang, small_hidden_size, device
    ):
        """Test encoder hidden state shape is (1, batch, hidden_size)."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 5
        input_tensor = torch.randint(
            0, sample_lang.n_words, (batch_size, seq_len), device=device
        )

        output, hidden = encoder(input_tensor)

        assert hidden.shape == (1, batch_size, small_hidden_size)

    def test_encoder_batch_processing(self, sample_lang, small_hidden_size, device):
        """Test encoder handles different batch sizes."""
        torch.manual_seed(42)
        encoder = EncoderRNN(sample_lang.n_words, small_hidden_size)
        encoder.to(device)
        encoder.eval()

        for batch_size in [1, 2, 4]:
            input_tensor = torch.randint(
                0, sample_lang.n_words, (batch_size, 5), device=device
            )
            output, hidden = encoder(input_tensor)
            assert output.shape[0] == batch_size
            assert hidden.shape[1] == batch_size


class TestBahdanauAttention:
    def test_attention_weights_sum_to_one(self, small_hidden_size, device):
        """Test that attention weights sum to 1 (softmax property)."""
        torch.manual_seed(42)
        attention = BahdanauAttention(small_hidden_size)
        attention.to(device)

        batch_size = 2
        seq_len = 5
        query = torch.randn(batch_size, 1, small_hidden_size, device=device)
        keys = torch.randn(batch_size, seq_len, small_hidden_size, device=device)

        context, weights = attention(query, keys)

        # Weights should sum to 1 along the sequence dimension
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_attention_context_shape(self, small_hidden_size, device):
        """Test context vector shape is (batch, 1, hidden_size)."""
        torch.manual_seed(42)
        attention = BahdanauAttention(small_hidden_size)
        attention.to(device)

        batch_size = 2
        seq_len = 5
        query = torch.randn(batch_size, 1, small_hidden_size, device=device)
        keys = torch.randn(batch_size, seq_len, small_hidden_size, device=device)

        context, weights = attention(query, keys)

        assert context.shape == (batch_size, 1, small_hidden_size)

    def test_attention_weights_shape(self, small_hidden_size, device):
        """Test attention weights shape is (batch, 1, seq_len)."""
        torch.manual_seed(42)
        attention = BahdanauAttention(small_hidden_size)
        attention.to(device)

        batch_size = 2
        seq_len = 5
        query = torch.randn(batch_size, 1, small_hidden_size, device=device)
        keys = torch.randn(batch_size, seq_len, small_hidden_size, device=device)

        context, weights = attention(query, keys)

        assert weights.shape == (batch_size, 1, seq_len)


class TestDecoderRNN:
    def test_decoder_output_shape(
        self, decoder, sample_lang, small_hidden_size, device
    ):
        """Test decoder output shape is (batch, MAX_LENGTH, vocab_size)."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 5

        encoder_outputs = torch.randn(
            batch_size, seq_len, small_hidden_size, device=device
        )
        encoder_hidden = torch.randn(1, batch_size, small_hidden_size, device=device)

        output, hidden, attn = decoder(encoder_outputs, encoder_hidden)

        assert output.shape == (batch_size, MAX_LENGTH, sample_lang.n_words)

    def test_decoder_teacher_forcing(
        self, decoder, sample_lang, small_hidden_size, device
    ):
        """Test decoder uses target tensor when provided (teacher forcing)."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 5

        encoder_outputs = torch.randn(
            batch_size, seq_len, small_hidden_size, device=device
        )
        encoder_hidden = torch.randn(1, batch_size, small_hidden_size, device=device)
        target_tensor = torch.randint(
            0, sample_lang.n_words, (batch_size, MAX_LENGTH), device=device
        )

        output, hidden, attn = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # Output should still have correct shape
        assert output.shape == (batch_size, MAX_LENGTH, sample_lang.n_words)

    def test_decoder_greedy(self, decoder, sample_lang, small_hidden_size, device):
        """Test decoder uses own predictions when no target provided."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 5

        encoder_outputs = torch.randn(
            batch_size, seq_len, small_hidden_size, device=device
        )
        encoder_hidden = torch.randn(1, batch_size, small_hidden_size, device=device)

        output, hidden, attn = decoder(encoder_outputs, encoder_hidden)

        # Should still produce output
        assert output.shape == (batch_size, MAX_LENGTH, sample_lang.n_words)
        # Attention should be None for basic decoder
        assert attn is None


class TestAttnDecoderRNN:
    def test_attn_decoder_output_shape(
        self, attn_decoder, sample_lang, small_hidden_size, device
    ):
        """Test attention decoder output shape."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 5

        encoder_outputs = torch.randn(
            batch_size, seq_len, small_hidden_size, device=device
        )
        encoder_hidden = torch.randn(1, batch_size, small_hidden_size, device=device)

        output, hidden, attn = attn_decoder(encoder_outputs, encoder_hidden)

        assert output.shape == (batch_size, MAX_LENGTH, sample_lang.n_words)

    def test_attn_decoder_returns_attention_weights(
        self, attn_decoder, small_hidden_size, device
    ):
        """Test that attention decoder returns attention weights."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 5

        encoder_outputs = torch.randn(
            batch_size, seq_len, small_hidden_size, device=device
        )
        encoder_hidden = torch.randn(1, batch_size, small_hidden_size, device=device)

        output, hidden, attn = attn_decoder(encoder_outputs, encoder_hidden)

        assert attn is not None

    def test_attn_decoder_attention_shape(
        self, attn_decoder, small_hidden_size, device
    ):
        """Test attention weights shape is (batch, MAX_LENGTH, seq_len)."""
        torch.manual_seed(42)
        batch_size = 2
        seq_len = 5

        encoder_outputs = torch.randn(
            batch_size, seq_len, small_hidden_size, device=device
        )
        encoder_hidden = torch.randn(1, batch_size, small_hidden_size, device=device)

        output, hidden, attn = attn_decoder(encoder_outputs, encoder_hidden)

        assert attn.shape == (batch_size, MAX_LENGTH, seq_len)
