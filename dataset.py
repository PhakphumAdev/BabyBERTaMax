from transformers import BertTokenizer
from datasets import load_dataset

class babyDataset:
    def __init__(self, filepath, tokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.dataset = load_dataset('text', data_files=filepath)
        # use babyberta tokenizer to tokenize the dataset
        self.tokenized_dataset = tokenizer.encode(self.dataset, add_special_tokens=False).tokens
