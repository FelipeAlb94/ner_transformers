import sys, os
from transformers import AutoTokenizer
import argparse
import random
random.seed(2)

file_dir = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(file_dir)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ner_transformers.conll import ConllDataset

def split_conll(filepath, train_split_size, outdir, shuffle=True):
    parsed_file = list(ConllDataset._read_file(filepath))
    filename, ext = os.path.splitext(os.path.basename(filepath))
    if shuffle:
        random.shuffle(parsed_file)
    
    train_size = int(len(parsed_file) * train_split_size)
    #Train set
    train_set = parsed_file[:train_size]
    trainset_filename = f"{filename}_train{ext}"
    trainset_outpath = os.path.join(outdir, trainset_filename)
    ConllDataset._write_file(train_set, trainset_outpath)
    #Test set
    test_set = parsed_file[train_size:]
    testset_filename = f"{filename}_test{ext}"
    testset_outpath = os.path.join(outdir, testset_filename)
    ConllDataset._write_file(test_set, testset_outpath)
    return trainset_filename, testset_filename

def preprocess(args):
    dataset, model_name_or_path, max_len, overlap, output_path = args.dataset, args.model_name_or_path, args.max_len,\
                                                                    args.overlap, args.output_path
    subword_len_counter = 0

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    max_len -= tokenizer.num_special_tokens_to_add()


    with open(dataset, "rt") as f_p, open(
            output_path, "w", encoding="utf8"
        ) as fw_p:

        lines = f_p.readlines()
        for i, line in enumerate(lines):
            line = line.rstrip()

            if not line or line.startswith("-DOCSTART-") or line == "" or line == "\n":
                fw_p.write(line + "\n")
                subword_len_counter = 0
                continue

            token = line.split(' ')[0]

            current_subwords_len = len(tokenizer.tokenize(token))

            # Token contains strange control characters like \x96 or \x95
            # Just filter out the complete line
            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                fw_p.write("\n")
                subword_len_counter = current_subwords_len
                for l in lines[i-overlap:i+1]:
                    fw_p.write(l.rstrip() + "\n")
                    subword_len_counter += len(tokenizer.tokenize(l.split(' ')[0]))
                continue
            
            fw_p.write(line + "\n")
            subword_len_counter += current_subwords_len

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args(raw_args)
    preprocess(args)

if __name__ == "__main__":
    split_conll('/opt/data/cvs_after_curadory/label_studio/export/project-16-at-2022-03-15-17-21-61656be3.conll', 0.7)

