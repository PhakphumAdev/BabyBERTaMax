from transformers import BertTokenizer
from datasets import load_dataset

class babyDataset:
    def __init__(self, filepath, tokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer
        raw_dataset = load_dataset('text', data_files=filepath)
        sentences = raw_dataset['train']['text']
        self.dataset = ' '.join(sentences)
        # use babyberta tokenizer to tokenize the dataset
        self.tokenized_dataset = tokenizer.encode(self.dataset, add_special_tokens=False).tokens
