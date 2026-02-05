import torch

from ml_translate.data import (
    Lang,
    EOS_token,
    unicodeToAscii,
    normalizeString,
    filterPair,
    filterPairs,
    indexesFromSentence,
    tensorFromSentence,
)


class TestLang:
    def test_lang_init(self):
        """Test initial state with SOS/EOS tokens."""
        lang = Lang("test")
        assert lang.name == "test"
        assert lang.n_words == 2
        assert lang.index2word[0] == "SOS"
        assert lang.index2word[1] == "EOS"
        assert lang.word2index == {}
        assert lang.word2count == {}

    def test_lang_add_word_new(self):
        """Test adding a new word increments n_words."""
        lang = Lang("test")
        lang.addWord("hello")
        assert lang.n_words == 3
        assert lang.word2index["hello"] == 2
        assert lang.word2count["hello"] == 1
        assert lang.index2word[2] == "hello"

    def test_lang_add_word_existing(self):
        """Test adding existing word increments count but not n_words."""
        lang = Lang("test")
        lang.addWord("hello")
        lang.addWord("hello")
        assert lang.n_words == 3  # Still 3, not 4
        assert lang.word2count["hello"] == 2

    def test_lang_add_sentence(self):
        """Test adding a sentence adds all words."""
        lang = Lang("test")
        lang.addSentence("hello world")
        assert lang.n_words == 4  # SOS, EOS, hello, world
        assert "hello" in lang.word2index
        assert "world" in lang.word2index


class TestUnicodeToAscii:
    def test_unicode_to_ascii_accents(self):
        """Test conversion of accented characters."""
        assert unicodeToAscii("café") == "cafe"
        assert unicodeToAscii("résumé") == "resume"

    def test_unicode_to_ascii_diacritics(self):
        """Test conversion of diacritics."""
        assert unicodeToAscii("naïve") == "naive"
        assert unicodeToAscii("coöperate") == "cooperate"

    def test_unicode_to_ascii_plain(self):
        """Test that plain ASCII is unchanged."""
        assert unicodeToAscii("hello") == "hello"
        assert unicodeToAscii("test123") == "test123"


class TestNormalizeString:
    def test_normalize_string_lowercase(self):
        """Test conversion to lowercase."""
        assert normalizeString("HELLO").startswith("hello")

    def test_normalize_string_punctuation_spacing(self):
        """Test spacing around punctuation."""
        result = normalizeString("hello!")
        assert "hello !" in result

    def test_normalize_string_removes_numbers(self):
        """Test that numbers are removed."""
        result = normalizeString("test123")
        assert "123" not in result
        assert "test" in result

    def test_normalize_string_strips_whitespace(self):
        """Test stripping of leading/trailing whitespace."""
        result = normalizeString("  hello  ")
        assert result == "hello"


class TestFilterPair:
    def test_filter_pair_valid(self):
        """Test that valid pairs pass the filter."""
        # Must have <10 words and second sentence starts with eng_prefixes
        pair = ["bonjour", "i am happy"]
        assert filterPair(pair) is True

    def test_filter_pair_too_long(self):
        """Test that pairs with >10 words fail."""
        long_sentence = " ".join(["word"] * 15)
        pair = [long_sentence, "i am happy"]
        assert filterPair(pair) is False

    def test_filter_pair_wrong_prefix(self):
        """Test that pairs without eng_prefixes fail."""
        pair = ["bonjour", "hello world"]  # doesn't start with eng_prefixes
        assert filterPair(pair) is False

    def test_filter_pair_both_conditions(self):
        """Test pair that fails both conditions."""
        long_sentence = " ".join(["word"] * 15)
        pair = [long_sentence, "hello world"]
        assert filterPair(pair) is False


class TestFilterPairs:
    def test_filter_pairs_mixed(self):
        """Test filtering a list of mixed pairs."""
        pairs = [
            ["bonjour", "i am happy"],  # valid
            ["salut", "hello world"],  # invalid prefix
            ["hi", "he is tall"],  # valid
        ]
        filtered = filterPairs(pairs)
        assert len(filtered) == 2
        assert ["bonjour", "i am happy"] in filtered
        assert ["hi", "he is tall"] in filtered

    def test_filter_pairs_empty(self):
        """Test filtering an empty list."""
        assert filterPairs([]) == []

    def test_filter_pairs_all_invalid(self):
        """Test filtering when all pairs are invalid."""
        pairs = [
            ["test", "hello world"],
            ["test", "goodbye world"],
        ]
        filtered = filterPairs(pairs)
        assert len(filtered) == 0


class TestIndexesFromSentence:
    def test_indexes_from_sentence(self):
        """Test conversion of sentence to indices."""
        lang = Lang("test")
        lang.addSentence("hello world")
        indexes = indexesFromSentence(lang, "hello world")
        assert indexes == [lang.word2index["hello"], lang.word2index["world"]]

    def test_indexes_from_sentence_single_word(self):
        """Test conversion of single word."""
        lang = Lang("test")
        lang.addWord("hello")
        indexes = indexesFromSentence(lang, "hello")
        assert indexes == [lang.word2index["hello"]]


class TestTensorFromSentence:
    def test_tensor_from_sentence_shape(self):
        """Test correct tensor shape."""
        lang = Lang("test")
        lang.addSentence("hello world")
        device = torch.device("cpu")
        tensor = tensorFromSentence(lang, "hello world", device)
        # Shape should be (1, sentence_length + 1) for EOS
        assert tensor.shape == (1, 3)  # "hello", "world", EOS

    def test_tensor_from_sentence_eos(self):
        """Test that EOS token is appended."""
        lang = Lang("test")
        lang.addSentence("hello world")
        device = torch.device("cpu")
        tensor = tensorFromSentence(lang, "hello world", device)
        # Last element should be EOS_token
        assert tensor[0, -1].item() == EOS_token

    def test_tensor_from_sentence_device(self):
        """Test that tensor is on correct device."""
        lang = Lang("test")
        lang.addWord("hello")
        device = torch.device("cpu")
        tensor = tensorFromSentence(lang, "hello", device)
        assert tensor.device.type == "cpu"
