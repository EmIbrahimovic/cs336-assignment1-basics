import regex as re
from typing import Iterable, Iterator, Generator, Any
from collections import defaultdict
import json


def _split_on_special_tokens(text: str, special_tokens=None) -> Generator[str, Any, None]:
    """
    Generator that splits up the text into stretches of special-token-free text and pure special token.
    Yields strings that are either spans of text without any special tokens or special tokens.
    """
    special_tokens_sorted = [] if not special_tokens else sorted(special_tokens, key=len, reverse=True)
    buffer = []
    i = 0
    while i < len(text):
        matching_tok = None
        for special_token in special_tokens_sorted:
            if text.startswith(special_token, i):
                matching_tok = special_token
                break

        if matching_tok:
            if buffer:
                yield "".join(buffer)
            yield matching_tok
            buffer.clear()
            i += len(matching_tok)
        else:
            buffer.append(text[i])
            i += 1

    if buffer:
        yield "".join(buffer)


def _pretokenize(text: str, special_tokens=None) -> list[str]:
    """
    Splits up the text into a list of pretokens - space or punctuation-separated strings.
    Isolates special tokens.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretokenized: list[str] = []
    for span in _split_on_special_tokens(text, special_tokens):
        if special_tokens and span in special_tokens:
            pretokenized.append(span)
        else:
            pretokenized.extend(re.findall(PAT, span))

    return pretokenized


def train_bpt_tokenizer(input_path, vocab_size, special_tokens):
    """
    :param input_path:
    :param vocab_size:
    :param special_tokens:
    :return: vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]
    """
    # Read file
    with open(input_path, 'r') as f:
        train_with_specials = f.read()

    encoded_special_toks = [token.encode('utf-8') for token in special_tokens]
    vocab = {**{c: bytes([c]) for c in range(256)},
             **dict(enumerate(encoded_special_toks, 256))}
    merges = []

    # Pre-tokenize
    pretokenized = _pretokenize(train_with_specials, special_tokens)
    train_tkn_cnt = defaultdict(int)
    for pretoken in pretokenized:
        if pretoken not in special_tokens:
            train_tkn_cnt[tuple(pretoken.encode('utf-8'))] += 1

    # Repeat merge loop until we saturate vocab, keeping track of merges.
    while len(vocab) < vocab_size:
        # TODO: avoid recomputing
        pairs_cnt = {}
        for word, cnt in train_tkn_cnt.items():
            for my_pair in zip(word, word[1:]):
                if my_pair not in pairs_cnt:
                    pairs_cnt[my_pair] = 0
                pairs_cnt[my_pair] += cnt

        # If all words have been reduced to a single token, quit.
        if not pairs_cnt:
            break

        # Identify the max-occurrence pair, breaking ties lexicographically
        # Append it to the vocabulary and merges
        # print(sorted(pairs_cnt.items()))
        pair_to_merge, max_cnt = max(pairs_cnt.items(), key=lambda kv: (kv[1], (vocab[kv[0][0]], vocab[
            kv[0][1]])))  # choose the max count. otherwise choose the max lex order
        # print(pair_to_merge, max_cnt)
        tok1, tok2 = int(pair_to_merge[0]), int(pair_to_merge[1])
        new_voc = len(vocab)
        vocab[new_voc] = vocab[tok1] + vocab[tok2]
        merges.append(tuple((vocab[tok1], vocab[tok2])))

        # TODO: maybe avoid recomputing
        train_tkn_cnt_new = {}
        for word, cnt in train_tkn_cnt.items():
            new_word = []
            # TODO: possible speedup
            skip_next = False
            for tok1, tok2 in zip(word, word[1:] + (-1,)):
                if skip_next:  # TODO: potentially remove this special token check
                    skip_next = False
                    continue

                if (tok1, tok2) == pair_to_merge:
                    new_word.append(new_voc)
                    skip_next = True
                else:
                    new_word.append(tok1)

            train_tkn_cnt_new[tuple(new_word)] = cnt

        train_tkn_cnt = train_tkn_cnt_new

    return vocab, merges


# ===========================================================
#                       Tokenizer
# ===========================================================

class Tokenizer:
    vocab: dict[int, bytes] = None
    inverse_vocab: dict[bytes, int] = None
    merges: list[tuple[bytes, bytes]] = None
    special_tokens: list[str] | None = None

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Parameters:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.inverse_vocab = {token: tok_id for tok_id, token in vocab.items()}
        self.merges = merges
        if special_tokens:
            self.special_tokens = special_tokens

            # Add special tokens to vocabulary and remember their encodings.
            for special_token in special_tokens:
                encoded_special_token = special_token.encode('utf-8')
                if encoded_special_token not in set(self.vocab.values()):
                    self.inverse_vocab[encoded_special_token] = len(self.vocab)
                    self.vocab[len(self.vocab)] = encoded_special_token

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the
        same format that your BPE training code output) and (optionally) a list of special tokens.

        Parameters:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(cleaned_line.split(" ")))

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        pretokenized_text: list[str] = _pretokenize(text, self.special_tokens)
        tokenized_text: list[int] = []

        for word in pretokenized_text:
            encoded_word: bytes = word.encode('utf-8')

            # Special tokens should be preserved
            if self.special_tokens and word in self.special_tokens:
                tokenized_text.append(self.inverse_vocab[encoded_word])
                continue

            curr_tokenization: list[int] = [self.inverse_vocab[bytes([c])] for c in encoded_word]
            next_tokenization: list[int] = []
            skip_next = False
            for merge in self.merges:
                for id1, id2 in zip(curr_tokenization, curr_tokenization[1:] + [-1]):
                    # We appended a merged token previously
                    if skip_next:
                        skip_next = False
                        continue

                    if id2 != -1 and (self.vocab[id1], self.vocab[id2]) == merge:
                        next_tokenization.append(self.inverse_vocab[merge[0] + merge[1]])
                        skip_next = True
                    else:
                        next_tokenization.append(id1)
                curr_tokenization = next_tokenization
                next_tokenization = []

            tokenized_text.extend(curr_tokenization)

        return tokenized_text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """

        def _encode_generator():
            for text_chunk in iterable:
                tokens = self.encode(text_chunk)
                yield from tokens

        return _encode_generator()

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        token_list = [self.vocab[id] for id in ids]
        all_bytes = b"".join(token_list)
        return all_bytes.decode('utf-8', errors='replace')
