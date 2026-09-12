from typing import Dict, List, Tuple


class BasicTokeniser:
    vocab: Dict[int, bytes]
    merges: Dict[Tuple[int, int], int]

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        assert vocab_size > 256, "Vocab should include at least all byte values"

    def encode(self, text: str) -> List[int]:
        utf8_bytes = list(text.encode("utf-8"))
        for pair, replacement in self.merges.items():
            new_bytes: List[int] = []
            i = 0
            while i < len(utf8_bytes):
                if (
                    i < len(utf8_bytes) - 1
                    and (utf8_bytes[i], utf8_bytes[i + 1]) == pair
                ):
                    new_bytes.append(replacement)
                    i += 2
                else:
                    new_bytes.append(utf8_bytes[i])
                    i += 1
            utf8_bytes = new_bytes
        return utf8_bytes

    def decode(self, tokens: List[int]):
        return b"".join([self.vocab[t] for t in tokens]).decode(
            "utf-8", errors="ignore"
        )

    def train(self, text: str) -> None:
        merges: Dict[Tuple[int, int], int] = {}
        utf8_bytes = list(text.encode("utf-8"))
        replacement_byte = 256
        num_merges = self.vocab_size - 256
        while num_merges > 0:
            pair_counts: Dict[Tuple[int, int], int] = {}
            for pair in zip(utf8_bytes, utf8_bytes[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            max_pair = max(pair_counts, key=lambda x: pair_counts[x])
            merges[max_pair] = replacement_byte
            new_bytes: List[int] = []
            i = 0
            while i < len(utf8_bytes):
                if (
                    i < len(utf8_bytes) - 1
                    and (utf8_bytes[i], utf8_bytes[i + 1]) == max_pair
                ):
                    new_bytes.append(replacement_byte)
                    i += 2
                else:
                    new_bytes.append(utf8_bytes[i])
                    i += 1
            replacement_byte += 1
            num_merges -= 1
            utf8_bytes = new_bytes
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = merges
        for (a, b), idx in merges.items():  # Iterates in the order of entry
            self.vocab[idx] = (
                self.vocab[a] + self.vocab[b]
            )  # Concatenate byte sequences
