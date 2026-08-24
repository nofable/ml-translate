# ml-translate

My attempt at building the original transformer from "Attention is all you need" to translate french to english and vice-versa.
<https://arxiv.org/pdf/1706.03762>

**Self-set rules:**

- Human coded, AI reviewed.
- Avoid looking at Annotated Transformer. Let my brain do the work.
- Avoid using 3P deps other than numpy.
- Use types

**Goals**

- Develop a deep understanding of the original transformer.
- Build strong intuition for training attention-based neural networks.

## Running Tests

Run all tests:

```bash
uv run pytest
```

## Jupyter & Jupytext

For setup i ran:

```bash
uv add jupyterlab jupytext
uv run jupyter labextension enable jupyterlab-jupytext
```

To run jupyter lab

```bash
uv run jupyter lab
```

Then right click on the notebook and open with notebook to make the jupytext render correctly.

## Classes from Annotated Transformer

- Encoder
- LayerNorm
- SublayerConnection
- EncoderLayer
- Decoder
- DecoderLayer
- attention
- MultiHeadAttention
- PositionWiseFeedForward
- Embeddings
- PositionalEncoding
