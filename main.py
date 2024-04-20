import random
import torch
from pathlib import Path
from transformers import RobertaForMaskedLM, RobertaConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from babyberta import load_tokenizer, Params
from babyberta.dataset import DataSet
from tokenizers import Tokenizer
from params import params

class Data:
    min_sentence_length = 3
    train_prob = 1.0  # probability that sentence is assigned to train split
    mask_symbol = '<mask>'
    pad_symbol = '<pad>'
    unk_symbol = '<unk>'
    bos_symbol = '<s>'
    eos_symbol = '</s>'
    roberta_symbols = [mask_symbol, pad_symbol, unk_symbol, bos_symbol, eos_symbol]

class BabyBERTaMaxTrainer:
    def __init__(self):
        self.tokenizer = self.loadTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #Configuration parameters from BabyBERTa
        config = RobertaConfig(vocab_size=len(self.tokenizer.get_vocab()),
                               pad_token_id=self.tokenizer.token_to_id(Data.pad_symbol),
                               bos_token_id=self.tokenizer.token_to_id(Data.bos_symbol),
                               eos_token_id=self.tokenizer.token_to_id(Data.eos_symbol),
                               return_dict=True,
                               is_decoder=False,
                               add_cross_attention=False,
                               layer_norm_eps=params.layer_norm_eps,
                               max_position_embeddings=params.max_input_length + 2,
                               hidden_size=params.hidden_size,
                               num_hidden_layers=params.num_layers,
                               num_attention_heads=params.num_attention_heads,
                               intermediate_size=params.intermediate_size,
                               intializer_range=params.initializer_range
                               )
        self.model = RobertaForMaskedLM(config=config)
        self.model.to(device)
    def loadTokenizer(self):
        tokenizer = Tokenizer.from_file(str("tokenizer/babyberta.json"))
        tokenizer.enable_truncation(max_length=params.max_input_length)
        return tokenizer
    def initializeModel(self):
        pass
    def trainModel(self):
        pass
    def saveModel(self):
        pass

if __name__ == "__main__":

