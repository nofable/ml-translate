import torch
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
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = torch.optim.AdamW(model.parameters(), lr=0.5, betas=(0.9, 0.98))

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
        inputs, inputs_pad_mask = tokenizer.encode(parts[0])
        expected_result, _ = tokenizer.encode(parts[1])
        outputs, outputs_pad_mask = tokenizer.encode(parts[1], right_shift=True)

        model.train()
        optim.zero_grad()

        logits = model.forward(
            inputs=inputs,
            outputs=outputs,
            inputs_pad_mask=inputs_pad_mask,
            outputs_pad_mask=outputs_pad_mask,
        )
        pad_mask_logits = torch.where(outputs_pad_mask == 0, 0.0, logits)
        loss = loss_fn(pad_mask_logits, expected_result.float())
        loss.backward()
        optim.step()
        print("expected_result", parts[1])
        print("actual_result", tokenizer.decode(logits))
        count += 1
