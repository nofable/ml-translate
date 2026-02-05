import logging
import random

import torch
from torch import Tensor

from ml_translate.data import Lang, tensorFromSentence, EOS_token
from ml_translate.model import EncoderRNN, DecoderRNN, AttnDecoderRNN

logger = logging.getLogger(__name__)


def evaluate(
    encoder: EncoderRNN,
    decoder: DecoderRNN | AttnDecoderRNN,
    sentence: str,
    input_lang: Lang,
    output_lang: Lang,
    device: torch.device,
) -> tuple[list[str], Tensor | None]:
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden
        )

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words: list[str] = []
        for idx in decoded_ids:
            idx_val = idx.item()
            if idx_val == EOS_token:
                decoded_words.append("<EOS>")
                break
            if idx_val not in output_lang.index2word:
                logger.warning(
                    "Unknown index %d not found in output vocabulary", idx_val
                )
                decoded_words.append("<UNK>")
            else:
                decoded_words.append(output_lang.index2word[idx_val])
    return decoded_words, decoder_attn


def evaluateRandomly(
    encoder: EncoderRNN,
    decoder: DecoderRNN | AttnDecoderRNN,
    input_lang: Lang,
    output_lang: Lang,
    pairs: list[list[str]],
    device: torch.device,
    n: int = 10,
) -> None:
    for i in range(n):
        pair = random.choice(pairs)
        logger.info("> %s", pair[0])
        logger.info("= %s", pair[1])
        output_words, _ = evaluate(
            encoder, decoder, pair[0], input_lang, output_lang, device
        )
        output_sentence = " ".join(output_words)
        logger.info("< %s", output_sentence)
        logger.info("")
