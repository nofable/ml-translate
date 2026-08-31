from pandas.io.common import file_exists
from tokenizers import Tokenizer
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

from ml_translate.collate import collate
from ml_translate.config import ENG_FRA_TEXT_FILE, PAD_TOKEN, TOKENIZER_FILE
from ml_translate.data_loader import TranslateDataLoader
from ml_translate.tokenizer import BytePairTokenizer
from ml_translate.transformer import Transformer


num_epochs = 2

if not file_exists(TOKENIZER_FILE):
    tokenizer = BytePairTokenizer()
    tokenizer.train(src_file=ENG_FRA_TEXT_FILE, out_file=TOKENIZER_FILE)

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

model = Transformer(
    d_model=512,
    max_seq_len=10_000,
    num_embeddings=10_000,
    n_layers=6,
    ff_d_hidden=2048,
    p_dropout=0.1,
)

PAD_ID = tokenizer.token_to_id(PAD_TOKEN)
assert PAD_ID is not None

loss_fn: CrossEntropyLoss = torch.nn.CrossEntropyLoss(
    label_smoothing=0.1, ignore_index=PAD_ID
)

optim = torch.optim.AdamW(model.parameters(), lr=0.5, betas=(0.9, 0.98))

dataloader = TranslateDataLoader()
train_dataloader, test_dataloader = dataloader.train_test_dataloaders()


def run_batch(model, tokenizer, x, y) -> tuple[Tensor, Tensor]:
    inputs, inputs_pad_mask = collate(x, tokenizer)
    outputs, outputs_pad_mask = collate(y, tokenizer)
    expected_result = outputs[:, 1:]
    shifted_right_outputs = outputs[:, :-1]
    shifted_right_outputs_mask = outputs_pad_mask[:, :-1]

    logits = model.forward(
        inputs=inputs,
        outputs=shifted_right_outputs,
        inputs_pad_mask=inputs_pad_mask,
        outputs_pad_mask=shifted_right_outputs_mask,
    )
    return logits, expected_result


for epoch in range(num_epochs):
    model.train()

    for train_x, train_y in train_dataloader:
        optim.zero_grad()
        logits, expected_result = run_batch(model, tokenizer, train_x, train_y)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            expected_result.reshape(-1),
        )
        loss.backward()
        optim.step()

    model.eval()
    total_loss = 0

    for test_x, test_y in test_dataloader:
        logits, expected_result = run_batch(model, tokenizer, test_x, test_y)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            expected_result.reshape(-1),
        )
        total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"epoch {epoch}: {avg_loss:.4f}")
