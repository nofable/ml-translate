from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader

from ml_translate.config import TOKENIZER_FILE, ENG_FRA_TEXT_FILE
from ml_translate.dataset import TabSeparatedLineDelimTranslationPairsDataset
from ml_translate.transformer import Transformer
from ml_translate.collate import collate


seq_len = 20
num_epochs = 10

tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

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

train_dataset = TabSeparatedLineDelimTranslationPairsDataset(filepath=ENG_FRA_TEXT_FILE)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)


for epoch in range(num_epochs):
    model.train()

    for inputs, outputs in train_dataloader:
        optim.zero_grad()

        # encode the inputs and outputs

        inputs, inputs_pad_mask = collate(inputs, tokenizer)
        outputs, outputs_pad_mask = collate(outputs, tokenizer)
        expected_result = outputs[:, :, :-1]

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

    model.eval()
    total_loss = 0
    for inputs, outputs in test_dataloader:
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"epoch {epoch}: {avg_loss:.4f}")
