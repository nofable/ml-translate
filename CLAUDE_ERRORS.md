- writing large code block with roll-your-own impl rather than using a third-party lib.
  - eg. implemented bleu score from scratch rather than using `from torchtext.data.metrics import bleu_score`
  - eg. implemented dataset split itself, rather than using sklearn's `train_test_split`
- didn't add the `.coverage` file to `.gitignore`
- puts in defaults without reasoning. eg. why did it choose `max_grad_norm` defaults to `1`?

