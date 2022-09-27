import sys, os
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from tqdm import tqdm
import torch
from pathlib import Path
from noronha.tools.shortcuts import model_path

nb_dir = os.path.abspath('')
BASE_DIR = os.path.dirname(nb_dir)
if not BASE_DIR in sys.path: sys.path.append(BASE_DIR)

from ner_transformers.extractor import Featurizer
from layoutxlm.utils import general

ner_tar_path = Path(model_path(file_name='ner_predictor.tar.gz', model='leia-ia2-ner', version='ner-consultas-v0'))
general.uncompress_tar(ner_tar_path, ner_tar_path.parent)
model_dir = [p for p in ner_tar_path.parent.glob('*') if p.is_dir()][0]

params = {'model_name': model_dir,
          'batch_size': 8,
          'max_seq_length': 512,
          'overlap': 5,
          'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

tokenizer = AutoTokenizer.from_pretrained(params['model_name'], model_max_length=params['max_seq_length'])
config = AutoConfig.from_pretrained(params['model_name'])
model =  AutoModelForTokenClassification.from_pretrained(params['model_name'], config=config).to(params['device'])
featurizer = Featurizer(tokenizer, config.id2label, params)

import torch.nn

act = torch.nn.Softmax(dim=-1)

def torch_delete(tensor, indices, dim=0):
    mask = torch.ones(tensor.shape[dim], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def predict(input_data, input_type = "text", output_type="default"):
    
    # faz o split, gera predições
    # e dá o join removendo os overlaps dos strides...
    
    if input_type.lower() == "file":
        paragraph = featurizer.extract_text(input_data)
    else:
        paragraph = input_data
        
    features = featurizer.get_features(paragraph)
    res = []
    
    for idx, batch in enumerate(features):
        
        ids = batch['ids'].unsqueeze(0).to(params['device'], dtype = torch.long)
        mask = batch['mask'].unsqueeze(0).to(params['device'], dtype = torch.long)
        
        outputs = model(input_ids=ids, attention_mask=mask)
        probs = act(outputs.logits)
        preds = torch.argmax(probs, dim=2)
        tokens = featurizer.tokenizer.convert_ids_to_tokens(ids[0])
        
        remove_token_ids = []
        remove_tokens = []
        new_tokens = []
        for idx_pred, (pred, token) in enumerate(zip(preds[0], tokens)):
            if token.startswith("##") or token in ['[CLS]', '[SEP]', '[PAD]']:
                remove_token_ids.append(idx_pred)
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_tokens.append(token)
        if len(remove_token_ids) > 0:
            preds = torch_delete(preds[0], remove_token_ids)
            probs = torch_delete(probs[0], remove_token_ids)
        
        # remove os overlaps !
        if idx > 0:
            preds = torch_delete(preds, [i for i in range(featurizer.overlap - 1)])
            probs = torch_delete(probs, [i for i in range(featurizer.overlap - 1)])
            
            del new_tokens[:featurizer.overlap - 1] #tokens[:49] -> tava gerando o bug cabuloso
        preds_to_labels = [featurizer.id2label[k.item()] for k in preds]
        res.append((preds_to_labels, new_tokens, probs.detach().numpy()))
        
    all_preds = []
    all_tokens = []
    all_probs = []
    for i in res:
        all_preds.extend(i[0])
        all_tokens.extend(i[1])
        all_probs.extend(i[2])
    
    if output_type.lower() == "dataframe":
        return pd.DataFrame({"token": all_tokens,
                             "classification": all_preds,
                             "probs": all_probs})
                             
    elif output_type.lower() == "default":
        return all_preds, all_tokens, all_probs