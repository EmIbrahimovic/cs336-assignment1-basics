import regex as re
from typing import Iterable, Iterator, Generator, Any
import jaxtyping
from jaxtyping import Integer
from collections import defaultdict
import json
import heapq
from functools import total_ordering


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


def _pretokenize_and_count(text_with_specials: str, special_tokens: list[bytes], train_tkn_cnt: dict[tuple[int, ...], int]):
    # Pre-tokenize
    pretokenized: list[str] = _pretokenize(text_with_specials, special_tokens)

    # Frequencies of each word (pretoken)
    for pretoken in pretokenized:
        if pretoken not in special_tokens:
            train_tkn_cnt[tuple(pretoken.encode('utf-8'))] += 1

def _apply_merge(train_tkn_cnt, pair_to_merge, new_voc):
    train_tkn_cnt_new = {}
    for word, cnt in train_tkn_cnt.items():
        new_word = []
        # TODO: possible speedup
        skip_next = False
        for tok1, tok2 in zip(word, word[1:] + (-1,)):
            if skip_next: 
                skip_next = False
                continue

            if (tok1, tok2) == pair_to_merge:
                new_word.append(new_voc)
                skip_next = True
            else:
                new_word.append(tok1)

        train_tkn_cnt_new[tuple(new_word)] = cnt

    return train_tkn_cnt_new

def _merge_pair_in_word(word: tuple[int, ...], pair_to_merge: tuple[int, int], new_voc: int) -> tuple[int, ...]:
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair_to_merge:
            new_word.append(new_voc)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    
    return tuple(new_word)

def _count_pretokens_streamed(input_path: str, special_tokens: list[bytes]) -> dict[tuple[int, ...], int]:
    train_tkn_cnt: dict[tuple[int, ...], int] = defaultdict(int)
    chunk_size = 10**6
    with open(input_path, 'r') as f:
        buffer: str = ""
        while True:
            chunk: str = f.read(chunk_size)
            if not chunk:  # If we've processed everything
                if buffer:
                    _pretokenize_and_count(buffer, special_tokens, train_tkn_cnt)
                break 
            
            buffer += chunk
            # Find last complete pretoken boundary (approximate with newline or space)
            last_boundary = max(buffer.rfind('\n'), buffer.rfind(' '))
            if last_boundary != -1:
                to_process = buffer[:last_boundary + 1]
                buffer = buffer[last_boundary + 1:]
                _pretokenize_and_count(to_process, special_tokens, train_tkn_cnt)

    return train_tkn_cnt

@total_ordering
class ReverseOrder:
    def __init__(self, value):
        self.value = value
    
    def __eq__(self, other):
        return self.value == other.value
    
    def __lt__(self, other):
        return self.value > other.value  # Reverse!

# Make streaming tokenizer.
def train_bpt_tokenizer(input_path, vocab_size, special_tokens):
    """
    :param input_path:
    :param vocab_size:
    :param special_tokens:
    :return: vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]
    """

    encoded_special_toks: list[bytes] = [token.encode('utf-8') for token in special_tokens]
    vocab: dict[int, bytes] = {**{c: bytes([c]) for c in range(256)},
             **dict(enumerate(encoded_special_toks, 256))}
    merges: list[tuple[bytes, bytes]] = []

    # Count all pretokens, reading the file chunk by chunk
    train_tkn_cnt: dict[tuple[int, ...], int] = _count_pretokens_streamed(input_path, special_tokens)
    
    # train_tkn_cnt should have a reasonable size, no need to chunk this.
    print(f"Found {len(train_tkn_cnt)} unique pretokens")

    pairs_cnt: dict[tuple[int, int], int] = defaultdict(int) # A pair is a tuple of token_ids
    words_for_pair: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for word_tuple, cnt in train_tkn_cnt.items():
        for tok_id_pair in zip(word_tuple, word_tuple[1:]):
            pairs_cnt[tok_id_pair] += cnt
            words_for_pair[tok_id_pair].add(word_tuple) # storing words here is fine cause we can access them easily 
    
    # Create a max heap to know the most frequent pair instantly
    heap: list[tuple[int, ReverseOrder, tuple[int, int]]] = []
    for tok_id_pair, cnt in pairs_cnt.items():
        tok1_bytes = vocab[tok_id_pair[0]]
        tok2_bytes = vocab[tok_id_pair[1]]
        heapq.heappush(heap, (-cnt, ReverseOrder((tok1_bytes, tok2_bytes)), tok_id_pair))

    print("train_tokenizer: pair counting complete")

    # Repeat merge loop until we saturate vocab, keeping track of merges.
    while len(vocab) < vocab_size:
        # Identify the max-occurrence pair, breaking ties lexicographically
        # Append it to the vocabulary and merges
        if not heap:
            print(f"No more pairs to merge. Stopping at vocab size {len(vocab)}")
            break

        # pair_to_merge: tuple[int, int]
        # tok1_bytes, tok2_bytes: bytes
        neg_cnt, _, pair_to_merge = heapq.heappop(heap)
        tok1_bytes = vocab[pair_to_merge[0]]
        tok2_bytes = vocab[pair_to_merge[1]]
        cnt = -neg_cnt

        if pair_to_merge not in pairs_cnt or pairs_cnt[pair_to_merge] != cnt:
            # Skip if this pair count is stale (was updated after being added to heap)
            continue
        if cnt == 0: # All words have been merged into single tokens
            break

        new_voc: int = len(vocab)
        vocab[new_voc] = tok1_bytes + tok2_bytes
        merges.append((tok1_bytes, tok2_bytes))

        if len(vocab) % 500 == 0:
            print(f"At vocab size {len(vocab)}, merging {tok1_bytes} and {tok2_bytes} with count {cnt}")

        # After the merge, some pairs be eliminated (.., tok_1) and (tok_2, ..)
        # Other pairs will be created (.., pair_to_merge), (pair_to_merge, ...)
        affected_pairs: set[tuple[int, int]] = set()

        words_to_update: list[tuple[int, ...]] = list(words_for_pair[pair_to_merge])
        for word_tuple in words_to_update:
            old_word: tuple[int, ...] = word_tuple
            word_cnt = train_tkn_cnt[old_word]

            new_word = _merge_pair_in_word(old_word, pair_to_merge, new_voc)
            # Removing the old word kills old pairs
            for old_pair in zip(old_word, old_word[1:]):
                affected_pairs.add(old_pair)
                pairs_cnt[old_pair] -= word_cnt
                if pairs_cnt[old_pair] == 0:
                    del pairs_cnt[old_pair]
                words_for_pair[old_pair].discard(old_word)
           
            # The new word will introduce new pairs
            for new_pair in zip(new_word, new_word[1:]):
                affected_pairs.add(new_pair)
                pairs_cnt[new_pair] += word_cnt
                words_for_pair[new_pair].add(new_word)
            
            del train_tkn_cnt[old_word]
            train_tkn_cnt[new_word] = word_cnt

        # After we've merged these tokens, we don't consider them in tracking
        if pair_to_merge in pairs_cnt:
            del pairs_cnt[pair_to_merge]
        if pair_to_merge in words_for_pair:
            del words_for_pair[pair_to_merge]
        
        # Finally, update the heop. Find all the pairs which we've fiddled with - new ones and old ones
        for affected_pair in affected_pairs:
            # If it's a new pair or an old pair that still exists in other words
            if affected_pair in pairs_cnt and pairs_cnt[affected_pair] > 0:
                tok1_bytes = vocab[affected_pair[0]]
                tok2_bytes = vocab[affected_pair[1]]
                heapq.heappush(heap, 
                            (-pairs_cnt[affected_pair], 
                            ReverseOrder((tok1_bytes, tok2_bytes)), 
                            affected_pair))


    return vocab, merges


# ===========================================================
#                       Tokenizer
# ===========================================================

class Tokenizer:
    vocab: dict[int, bytes]
    inverse_vocab: dict[bytes, int]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str]

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Parameters:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vocab = vocab

        # ENSURE all 256 bytes are in vocab
        for i in range(256):
            if i not in vocab:
                vocab[i] = bytes([i])

        self.inverse_vocab = {token: tok_id for tok_id, token in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        # For each pair of tokens that we need to merge, remember when they need
        self.merge_priority = {}
        for priority, (tok1_bytes, tok2_bytes) in enumerate(merges):
            merged_bytes = tok1_bytes + tok2_bytes
            if merged_bytes in self.inverse_vocab:
                tok1_id = self.inverse_vocab[tok1_bytes]
                tok2_id = self.inverse_vocab[tok2_bytes]
                merged_id = self.inverse_vocab[merged_bytes]
                self.merge_priority[(tok1_id, tok2_id)] = (merged_id, priority)

        # Add special tokens to vocabulary and remember their encodings.
        for special_token in self.special_tokens:
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

    # Add priority queue to encode
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        pretokenized_text: list[str] = _pretokenize(text, self.special_tokens)
        tokenized_text: list[int] = []

        for word in pretokenized_text:
            encoded_word: bytes = word.encode('utf-8')

            # Special tokens should be preserved
            if word in self.special_tokens:
                tokenized_text.append(self.inverse_vocab[encoded_word])
                continue

            tokens: list[int] = [self.inverse_vocab[bytes([c])] for c in encoded_word]
           
            while len(tokens) > 1:
                to_merge_idx = None
                highest_priority = float('inf')

                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merge_priority:
                        merged_id, priority = self.merge_priority[pair]
                        if priority < highest_priority:
                            highest_priority = priority
                            to_merge_idx = i
                
                if to_merge_idx is None:
                    break
                
                pair = (tokens[to_merge_idx], tokens[to_merge_idx + 1])
                merged_id, _ = self.merge_priority[pair]
                tokens = tokens[:to_merge_idx] + [merged_id] + tokens[to_merge_idx + 2:]
                
            tokenized_text.extend(tokens)

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
