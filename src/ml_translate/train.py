import re
from ml_translate.transformer import Transformer
from ml_translate.tokenizer import CharacterTokenizer


with open("data/eng-fra.txt") as file:
    seq_len = 20

    tokenizer = CharacterTokenizer(seq_len=seq_len)
    model = Transformer(
        d_model=512,
        seq_len=seq_len,
        num_embeddings=100,
        n_layers=6,
        ff_d_hidden=2048,
        p_dropout=0.1,
    )

    count = 0
    max_lines = 10

    for line in file:
        if count >= max_lines:
            break
        clean_line = line.strip()
        # Replaces NNBSP, which is used before punctuation in french
        clean_line = re.sub(r"\u202f", "", clean_line)
        parts = clean_line.split("\t", maxsplit=1)
        assert len(parts) == 2

        # build out the token set
        tokenizer.ingest(parts[0])
        tokenizer.ingest(parts[1])

        # encode the inputs and outputs
        inputs = tokenizer.encode(parts[0])
        expected_result = tokenizer.encode(parts[1])
        outputs = tokenizer.encode(parts[1], right_shift=True)
        result = model.forward(inputs, outputs)
        print("expected_result", parts[1])
        print("actual_result", tokenizer.decode(result))
        count += 1
