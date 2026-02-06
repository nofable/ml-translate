- writing large code block with roll-your-own impl rather than using a third-party lib.
  - eg. implemented bleu score from scratch rather than using `from torchtext.data.metrics import bleu_score`
  - eg. implemented dataset split itself, rather than using sklearn's `train_test_split`
- didn't add the `.coverage` file to `.gitignore`
- puts in defaults without reasoning. eg. why did it choose `max_grad_norm` defaults to `1`?
- Claude introduced a pre-trained embedding, but then added a trainable linear projection layer.
  - will the projection you have introduced need training since it's a new nn.Linear? doesn't that detract from this being a pretrained embedding?
  ⎿  Found 5 new diagnostic issues in 1 file (ctrl+o to expand)
  ⏺ You're right - the projection layer adds trainable parameters, which dilutes the benefit of pre-trained embeddings.

    The cleanest solutions:

    1. Match hidden_size to GloVe dimension - Use hidden_size=100 or hidden_size=300 (GloVe 6B supports 50, 100, 200, 300)
    2. Remove projection entirely - Only allow pre-trained embeddings when dimensions match

    Option 1 is simplest - just change your model's hidden_size to 100 or 300. The current 128 is arbitrary anyway.

    Want me to:
    - Remove the projection layer and require matching dimensions, or
    - Keep it simple and just use GloVe with hidden_size=100 (no projection needed)?

