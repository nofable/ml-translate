from pandas.io.common import file_exists
from tokenizers import Tokenizer
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split

from ml_translate.collate import collate
from ml_translate.config import ENG_FRA_TEXT_FILE, TOKENIZER_FILE
from ml_translate.dataset import TabSeparatedLineDelimTranslationPairsDataset
from ml_translate.tokenizer import BytePairTokenizer
from ml_translate.transformer import Transformer


seq_len = 20
num_epochs = 10

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

loss_fn: CrossEntropyLoss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optim = torch.optim.AdamW(model.parameters(), lr=0.5, betas=(0.9, 0.98))

full_dataset = TabSeparatedLineDelimTranslationPairsDataset(filepath=ENG_FRA_TEXT_FILE)
g = torch.Generator().manual_seed(42)
train_ds, test_ds = random_split(full_dataset, [0.8, 0.2], generator=g)
train_dataloader = DataLoader(train_ds, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=10)


for epoch in range(num_epochs):
    model.train()

    for train_x, train_y in train_dataloader:
        optim.zero_grad()

        inputs, inputs_pad_mask, _ = collate(train_x, tokenizer)
        outputs, outputs_pad_mask, outputs_causal_mask = collate(train_y, tokenizer)
        expected_result = outputs[:, :-1]
        shifted_right_outputs = outputs[:, 1:]
        shifted_right_outputs_mask = outputs_pad_mask[:, 1:]
        shifted_right_outputs_causal_mask = outputs_causal_mask[:, 1:]

        logits = model.forward(
            inputs=inputs,
            outputs=shifted_right_outputs,
            inputs_pad_mask=inputs_pad_mask,
            outputs_pad_mask=shifted_right_outputs_mask,
            outputs_causal_mask=shifted_right_outputs_causal_mask,
        )
        pad_mask_logits = torch.where(outputs_pad_mask == 0, 0.0, logits)
        loss = loss_fn(pad_mask_logits, expected_result.float())
        loss.backward()
        optim.step()

    model.eval()
    total_loss = 0

    for test_x, test_y in test_dataloader:
        inputs, inputs_pad_mask, _ = collate(test_x, tokenizer)
        outputs, outputs_pad_mask, outputs_causal_mask = collate(test_y, tokenizer)
        expected_result = outputs[:, :, :-1]
        shifted_right_outputs = outputs[:, :, 1:]
        shifted_right_outputs_mask = outputs_pad_mask[:, :, 1:]
        logits = model.forward(
            inputs=inputs,
            outputs=shifted_right_outputs,
            inputs_pad_mask=inputs_pad_mask,
            outputs_pad_mask=shifted_right_outputs_mask,
            outputs_causal_mask=outputs_causal_mask,
        )
        loss = loss_fn(logits, expected_result.float())
        total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"epoch {epoch}: {avg_loss:.4f}")
