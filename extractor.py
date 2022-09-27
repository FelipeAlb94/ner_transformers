import os, sys
import torch
import pandas as pd
import math

file_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(file_dir)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
        
from lib.base_classes import AttrDict
class Featurizer(AttrDict):

    def __init__(self, tokenizer, id2label, *args, **kwargs):

        super(Featurizer, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.id2label = id2label #{v: k for v, k in enumerate(list(self.unique_labels))}
        self.label2color = {'I-professional': 'blue',
                       'I-education': 'yellow',
                       'I-language': 'red',
                       'I-header': 'orange',
                       'I-aditional': 'pink',
                       'I-Skills': 'maroon',
                       'I-Summary': 'aqua',
                       'B-education': 'purple',
                       'B-Skills': 'green',
                       'B-aditional': 'navy',
                       'B-language': 'silver',
                       'B-Summary': 'teal',
                       'B-header': 'black',
                       'B-professional': 'fuchsia',

                        'O': 'olive'} # todo build property based on id2label


        
    def extract_text(self, file_path):
        import textract
        
        try:
            if file_path.endswith(".pdf"):
                text = textract.process(file_path, method='tesseract', language='spa')
            else:
                text = textract.process(file_path)
                text = text.decode('utf_8')

        except Exception as e:
            print(file_path, e)
            return None
        return text
    
    def split_long_paragraphs(self, paragraph, split_size):
        stride = self.overlap
        l_series_paragraph = []
        l_input_ids = self.tokenizer.encode(paragraph)

        # tem que considerar o chorim, que só pode acontecer se n_splits > 1 !
        if len(l_input_ids) > split_size:
            n_splits = (len(l_input_ids) + stride) / split_size
        else:
            n_splits = 1
            
        n_splits = math.ceil(n_splits)
        
        overlap = []
        for s in range(n_splits):
            if s == 0:
                
                # if: caso apenas 1 chunk: retorna o texto tokenizado
                # else: caso contrário, retorna o primeiro chunk cheio
                if s == n_splits - 1:
                    s = self.tokenizer.decode(l_input_ids, skip_special_tokens=True)
                else:
                    s = self.tokenizer.decode(l_input_ids[:split_size], skip_special_tokens=True)
                l_series_paragraph.append(pd.Series(dict(paragraph_id=0, paragraph=s)))


            elif s == n_splits - 1:
                start = split_size*s - stride
                s = self.tokenizer.decode(l_input_ids[start:], skip_special_tokens=True)
                l_series_paragraph.append(pd.Series(dict(paragraph_id=0, paragraph=s)))

            # se s está no meio    
            else:
                start = split_size*s - stride
                end = start + split_size
                s = self.tokenizer.decode(l_input_ids[start:end], skip_special_tokens=True)
                l_series_paragraph.append(pd.Series(dict(paragraph_id=0, paragraph=s)))

        return l_series_paragraph


    def convert_example_to_features(self, words):

        tokens = []
        
        all_tokens = self.tokenizer.tokenize(words)
        tokens.extend(all_tokens)

        # Truncation: account for [CLS] and [SEP] with "- 2". 
        special_tokens_count = 2 
        if len(tokens) > self.max_seq_length - special_tokens_count:
            tokens = tokens[: (self.max_seq_length - special_tokens_count)]

        # add [SEP] token, with corresponding token boxes and actual boxes
        tokens += [self.tokenizer.sep_token]

        # next: [CLS] token
        tokens = [self.tokenizer.cls_token] + tokens

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        input_mask += [0] * padding_length

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length

        return {"ids": torch.tensor(input_ids), 
                "mask": torch.tensor(input_mask)}
    
    def get_features(self, paragraph):
        
        sub_paragraphs = self.split_long_paragraphs(paragraph, split_size=self.max_seq_length)
        
        return [self.convert_example_to_features(sub_p['paragraph']) for sub_p in sub_paragraphs]