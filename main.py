import random
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from tokenizers import Tokenizer
from params import params
from dataset import babyDataset
from datasets import concatenate_datasets
import sys

class Data:
    min_sentence_length = 3
    train_prob = 1.0  # probability that sentence is assigned to train split
    mask_symbol = '<mask>'
    pad_symbol = '<pad>'
    unk_symbol = '<unk>'
    bos_symbol = '<s>'
    eos_symbol = '</s>'
    roberta_symbols = [mask_symbol, pad_symbol, unk_symbol, bos_symbol, eos_symbol]


class BabyBERTaMax:
    def __init__(self, curriculum=True):
        self.curriculum = curriculum
        self.tokenizer = self.loadTokenizer()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #Configuration parameters from BabyBERTa
        config = RobertaConfig(vocab_size=self.tokenizer.vocab_size,
                           hidden_size=params.hidden_size,
                           num_hidden_layers=params.num_layers,
                           num_attention_heads=params.num_attention_heads,
                           intermediate_size=params.intermediate_size,
                           initializer_range=params.initializer_range,
                           )
        self.model = RobertaForMaskedLM(config=config)
        self.train, self.test, self.dev = self.loadDataset(curriculum=curriculum)
        self.model.to(device)
        print('Number of parameters: {:,}'.format(self.model.num_parameters()), flush=True)
    def loadTokenizer(self):
        tokenizer = RobertaTokenizerFast(vocab_file=None,
                                     merges_file=None,
                                     tokenizer_file=str('tokenizer/babyberta.json'),
                                     )

        return tokenizer
    def loadDataset(self,curriculum=True):
        #train
        if curriculum:
            # if do curriculum learning, load 10M dataset in order
            dataset_order = [f"{corpus}.train" for corpus in params.corpora]  # add .train to each corpus in params.corpora
            dataset_paths = [Path("dataset/train_10M") / dataset for dataset in dataset_order]
            dataset_train = [babyDataset(str(dataset_path), self.tokenizer) for dataset_path in dataset_paths]
        else:
            # random
            dataset_order = random.sample(params.corpora, len(params.corpora))
            dataset_paths = [Path("dataset/train_10M") / f"{corpus}.train" for corpus in dataset_order]
            dataset_train = [babyDataset(str(dataset_path), self.tokenizer) for dataset_path in dataset_paths]

        #test
        dataset_order = [f"{corpus}.test" for corpus in params.corpora]
        dataset_paths = [Path("dataset/test") / dataset for dataset in dataset_order]
        dataset_test = [babyDataset(str(dataset_path), self.tokenizer) for dataset_path in dataset_paths]
        #dev
        dataset_order = [f"{corpus}.dev" for corpus in params.corpora]
        dataset_paths = [Path("dataset/dev") / dataset for dataset in dataset_order]
        dataset_dev = [babyDataset(str(dataset_path), self.tokenizer) for dataset_path in dataset_paths]
        return dataset_train, dataset_test, dataset_dev
    
    def trainModel(self):
        
        #prepare train dataset
        if len(self.train) > 1:
            # Ensure you are accessing the 'train' split of each dataset before concatenation
            combined_train_dataset = concatenate_datasets([ds.tokenized_dataset['train'] for ds in self.train])
        else:
            # Access the 'train' split of the single dataset
            combined_train_dataset = self.train[0].tokenized_dataset['train']

        # Data collator without masking for training
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=params.mask_probability)

        # Data collator with masking for validation and testing
        seed=77
        #adjust from official implementation huggingface
        training_args = TrainingArguments(
            output_dir="saved_model/babyberta_max_curriculum" if self.curriculum else "saved_model/babyberta_max_random",
            overwrite_output_dir=True,
            do_train=True,
            do_eval=False,
            do_predict=False,
            max_steps=160_000,
            per_device_train_batch_size=params.batch_size,
            warmup_steps=params.num_warmup_steps,
            seed=seed,
            learning_rate=params.lr,
            logging_dir="logs/babyberta_max_curriculum" if self.curriculum else "logs/babyberta_max_random",
            logging_steps=1000,
            save_steps=40_000,)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=combined_train_dataset,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )        
        trainer.train()
        trainer.save_model()

if __name__ == "__main__":
    # Check if curriculum argument is provided
    if len(sys.argv) > 1:
        curriculum_arg = sys.argv[1].lower()
        if curriculum_arg == "true":
            curriculum = True
        elif curriculum_arg == "false":
            curriculum = False
        else:
            print("Invalid curriculum argument. Please provide 'true' or 'false'.")
            sys.exit(1)
    else:
        print("Curriculum argument not provided. Please provide 'true' or 'false'.")
        sys.exit(1)
    
    # Init babyBERTa
    babyBERTa = BabyBERTaMax(curriculum=curriculum)
    babyBERTa.trainModel()