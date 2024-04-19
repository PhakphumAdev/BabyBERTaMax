#from transformers import RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments
#from tokenizers import ByteLevelBPETokenizer
#from torch.utils.data import Dataset
#from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

class BabyBERTaMaxTrainer:
    def __init__(self):
        pass
    def initializeModel(self):
        pass
    def trainModel(self):
        pass
    def saveModel(self):
        pass

if __name__ == "__main__":
    #Configuration parameters from BabyBERTa
    #config = RobertaConfig()
    #Load tokenizer from BabyBERTa
    tokenizer = Tokenizer.from_file(str("tokenizer/babyberta.json"))
    tokenizer.enable_truncation(max_length=128)
    vocabSize = len(tokenizer.get_vocab())
