import torch
from torch.utils.data import Dataset

class ConllDataset(Dataset):
    def __init__(self, params, file_path, tokenizer):
        
        parsed_file = list(self._read_file(file_path))
        self.len = len(parsed_file)
        self.sentences = [sentence['words'] for sentence in parsed_file]
        self.labels = [sentence['ner_tags'] for sentence in parsed_file]
        self.unique_labels = list(set([label for sent_labels in self.labels for label in sent_labels]))
        self.tokenizer = tokenizer
        self.max_len = params['max_seq_len']
        
    def __getitem__(self, index):
        
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.sentences[index]
        word_labels = self.labels[index]
        tokenized_sentence, labels = self.tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
        
        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        labels.insert(0, "O") # add outside label for [CLS] token
        labels.insert(-1, "O") # add outside label for [SEP] token

        # step 3: truncating/padding
        if (len(tokenized_sentence) > self.max_len):
          # truncate
          tokenized_sentence = tokenized_sentence[:self.max_len]
          labels = labels[:self.max_len]
        else:
          # pad
          tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(self.max_len - len(tokenized_sentence))]
          labels = labels + ["O" for _ in range(self.max_len - len(labels))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        
        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [self.label2id[label] for label in labels]
        
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

    @property
    def label2id(self):
        return {k: v for v, k in enumerate(list(self.unique_labels))}

    @property
    def id2label(self):
        return {v: k for v, k in enumerate(list(self.unique_labels))}
    
    @staticmethod
    def _read_file(filepath):
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            words = []
            pos_tags = []
            chunk_tags = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        yield {
                            "id": str(guid),
                            "words": words,
                            "pos_tags": pos_tags,
                            "chunk_tags": chunk_tags,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        words = []
                        pos_tags = []
                        chunk_tags = []
                        ner_tags = []
                else:
                    # conll2003 tokens are space separated
                    splits = line.split(" ")
                    words.append(splits[0])
                    pos_tags.append(splits[1])
                    chunk_tags.append(splits[2])
                    ner_tags.append(splits[3].rstrip())
            # last example
            yield {
                "id": str(guid),
                "words": words,
                "pos_tags": pos_tags,
                "chunk_tags": chunk_tags,
                "ner_tags": ner_tags,
            }
    
    @staticmethod
    def _write_file(parsed_conll, filepath):
        with open(filepath, 'wt') as f:
            f.write('-DOCSTART- -X- O\n')
            for sentence in parsed_conll:
                for word, pos_tag, chunk_tag, ner_tag in zip(sentence['words'], sentence['pos_tags'], \
                                                                sentence['chunk_tags'], sentence['ner_tags']):
                    f.write(f'{word} {pos_tag} {chunk_tag} {ner_tag}\n')
                f.write('\n')
    
    @staticmethod
    def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
        """
        Word piece tokenization makes it difficult to match word labels
        back up with individual word pieces. This function tokenizes each
        word one at a time so that it is easier to preserve the correct
        label for each subword. It is, of course, a bit slower in processing
        time, but it will help our model achieve higher accuracy.
        """

        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):

            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels