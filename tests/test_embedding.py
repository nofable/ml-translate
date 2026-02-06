import torch

from ml_translate.embedding import PretrainedEmbedding


class TestPretrainedEmbedding:
    def test_pretrained_embedding_random_init(self):
        """Test PretrainedEmbedding with random initialization."""
        embed = PretrainedEmbedding(num_embeddings=100, embedding_dim=50)

        x = torch.tensor([[1, 2, 3]])
        output = embed(x)

        assert output.shape == (1, 3, 50)

    def test_pretrained_embedding_with_weights(self):
        """Test PretrainedEmbedding with pre-trained weights."""
        weights = torch.randn(100, 50)
        embed = PretrainedEmbedding(
            num_embeddings=100,
            embedding_dim=50,
            pretrained_weights=weights,
        )

        x = torch.tensor([[0]])
        output = embed(x)

        assert torch.allclose(output.squeeze(), weights[0])

    def test_pretrained_embedding_frozen(self):
        """Test that frozen embeddings don't have gradients."""
        weights = torch.randn(100, 50)
        embed = PretrainedEmbedding(
            num_embeddings=100,
            embedding_dim=50,
            pretrained_weights=weights,
            freeze=True,
        )

        assert embed.embedding.weight.requires_grad is False

    def test_pretrained_embedding_trainable(self):
        """Test that unfrozen embeddings have gradients."""
        weights = torch.randn(100, 50)
        embed = PretrainedEmbedding(
            num_embeddings=100,
            embedding_dim=50,
            pretrained_weights=weights,
            freeze=False,
        )

        assert embed.embedding.weight.requires_grad is True
