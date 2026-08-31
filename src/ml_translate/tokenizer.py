from pandas.io.common import file_exists
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE

from ml_translate.config import TOKENIZER_FILE


class BytePairTokenizer:
    def __init__(self, src_file: str):
        if file_exists(TOKENIZER_FILE):
            return

        self.bytePairTokenizer = Tokenizer(BPE())
        # add space before first word to make it more like other words
        self.bytePairTokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=True
        )
        self.bytePairTokenizer.decoder = decoders.ByteLevel()
        self.bytePairTokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        self.trainer = trainers.BpeTrainer(
            vocab_size=10000,
            min_frequency=2,  # how many times a byte pair needs to be seen before being considered as a token
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[pad]", "[bos]", "[eos]"],
        )

        self.bytePairTokenizer.train([src_file], trainer=self.trainer)
        self.bytePairTokenizer.save(TOKENIZER_FILE, pretty=True)

    def getTokenizer(self):
        return Tokenizer.from_file(TOKENIZER_FILE)
