class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        vocab = []
        
        for word in texts: 
            vocab.extend(word.split())
        vocab = sorted(list(set(vocab)))
        for i, voc in enumerate(vocab):
            self.word_to_id[voc] = i + 4
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.id_to_word = {value : key for key, value in self.word_to_id.items()}
        self.vocab_size = len(vocab) + 4
        
    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        entire_text = text.lower().split()
        encoded = []
        for word in entire_text:
            if word in self.word_to_id:
                encoded.append(self.word_to_id.get(word))
            else:
                encoded.append(self.word_to_id.get(self.unk_token))
        return encoded

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        decoded = []
        for id in ids:
            decoded.append(self.id_to_word.get(id, self.unk_token))
        return " ".join(decoded)
        