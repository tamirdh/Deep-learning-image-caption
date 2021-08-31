import os
from collections import Counter
import spacy


class Vocabulary:
    spacy_eng = spacy.load("en_core_web_sm")

    def __init__(self, freq_threshold):
        # setting the pre-reserved tokens int to string tokens
        # PAD- padding symbol
        # SOS- Start of Sentence
        # EOS- end of sentence
        # UNK- unknown word (unknown\ below threshold)
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        # string to int tokens
        # its reverse dict self.itos
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in Vocabulary.spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for index, sentence in enumerate(sentence_list):

            for word in self.tokenize(sentence):
                frequencies[word] += 1

                # add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
<<<<<<< HEAD
                    if idx > 0 and idx % 1000 == 0:
                        print(f"Added {idx} words to vocab")
                    idx += 1
            if index > 0 and index % 1000 == 0:
=======
                    if idx > 0 and idx % 10000 == 0:
                        print(f"Added {idx} words to vocab")
                    idx += 1
            if index > 0 and index % 100000 == 0:
>>>>>>> origin/master
                print(f"Iterated {index} sentences")

        print(f"Done, added {idx-1} words to vocabulary")

    def numericalize(self, text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        result = [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                  for token in tokenized_text]
        return result
