import logging
import random

import torch
from torch import Tensor
from torchtext.data.metrics import bleu_score

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


def evaluate_bleu(
    encoder: EncoderRNN,
    decoder: DecoderRNN | AttnDecoderRNN,
    pairs: list[list[str]],
    input_lang: Lang,
    output_lang: Lang,
    device: torch.device,
    max_n: int = 4,
) -> float:
    """Evaluate model on pairs and compute corpus BLEU score.

    Uses torchtext.data.metrics.bleu_score for calculation.

    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        pairs: List of [input, reference] sentence pairs.
        input_lang: Input language vocabulary.
        output_lang: Output language vocabulary.
        device: Device to run evaluation on.
        max_n: Maximum n-gram order for BLEU (default 4).

    Returns:
        Corpus BLEU score between 0 and 1.
    """
    encoder.eval()
    decoder.eval()

    candidates: list[list[str]] = []
    references: list[list[list[str]]] = []

    for pair in pairs:
        input_sentence, reference_sentence = pair[0], pair[1]

        # Get model output
        output_words, _ = evaluate(
            encoder, decoder, input_sentence, input_lang, output_lang, device
        )

        # Remove <EOS> token from hypothesis if present
        if output_words and output_words[-1] == "<EOS>":
            output_words = output_words[:-1]

        # Tokenize reference (torchtext expects list of possible references per candidate)
        reference_words = reference_sentence.split()

        candidates.append(output_words)
        references.append([reference_words])  # Single reference per candidate

    # Calculate BLEU using torchtext with uniform weights
    weights = [1.0 / max_n] * max_n
    score = bleu_score(candidates, references, max_n=max_n, weights=weights)

    logger.info("BLEU-%d score: %.4f", max_n, score)
    return score
