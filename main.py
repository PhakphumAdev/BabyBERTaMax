import random
import torch
import torch.utils.data.Dataloader as DataLoader
from pathlib import Path
from transformers import RobertaForMaskedLM, RobertaConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from tokenizers import Tokenizer
from params import params
from dataset import babyDataset
class Data:
    min_sentence_length = 3
    train_prob = 1.0  # probability that sentence is assigned to train split
    mask_symbol = '<mask>'
    pad_symbol = '<pad>'
    unk_symbol = '<unk>'
    bos_symbol = '<s>'
    eos_symbol = '</s>'
    roberta_symbols = [mask_symbol, pad_symbol, unk_symbol, bos_symbol, eos_symbol]

class babyBERTaTrainer(Trainer):
    def __init__(self, *args, train_dataset, eval_dataset, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_collator_train = train_dataset
        self.data_collator_eval = eval_dataset
    def get_train_dataloader(self):
        """Use non-masking data collator for training"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator_train,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Use masking data collator for evaluation"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator_eval,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

class BabyBERTaMax:
    def __init__(self):
        self.tokenizer = self.loadTokenizer()
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
        self.train, self.test, self.dev = self.loadDataset()
        self.model.to(device)
        print('Number of parameters: {:,}'.format(self.model.num_parameters()), flush=True)
    def loadTokenizer(self):
        tokenizer = Tokenizer.from_file(str("tokenizer/babyberta.json"))
        tokenizer.enable_truncation(max_length=params.max_input_length)
        return tokenizer
    def loadDataset(self):
        #train
        dataset_order = [f"{corpus}.train" for corpus in params.corpora]  # add .train to each corpus in params.corpora
        dataset_paths = [Path("/dataset/train_10M") / dataset for dataset in dataset_order]
        dataset_train = [babyDataset(str(dataset_path), self.tokenizer) for dataset_path in dataset_paths]
        #test
        dataset_order = [f"{corpus}.test" for corpus in params.corpora]
        dataset_paths = [Path("/dataset/test") / dataset for dataset in dataset_order]
        dataset_test = [babyDataset(str(dataset_path), self.tokenizer) for dataset_path in dataset_paths]
        #dev
        dataset_order = [f"{corpus}.dev" for corpus in params.corpora]
        dataset_paths = [Path("/dataset/dev") / dataset for dataset in dataset_order]
        dataset_dev = [babyDataset(str(dataset_path), self.tokenizer) for dataset_path in dataset_paths]
        return dataset_train, dataset_test, dataset_dev
    
    def trainModel(self):

        # Data collator without masking for training
        data_collator_train = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Data collator with masking for validation and testing
        data_collator_eval = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        training_args = TrainingArguments(
            output_dir="saved_model/babyberta_max",
            overwrite_output_dir=True,
            num_train_epochs=params.num_epochs,
            per_device_train_batch_size=params.batch_size,
            warmup_steps=params.num_warmup_steps,
            learning_rate=params.lr,
            weight_decay=params.weight_decay,
            save_strategy="epoch",)
        
        trainer = babyBERTaTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train,
            eval_dataset=self.dev,
            data_collator_train=data_collator_train,
            data_collator_eval=data_collator_eval
        )
        trainer.train()
    def saveModel(self):
        pass

if __name__ == "__main__":
    #init babyBERTa
    babyBERTa = BabyBERTaMax()
    babyBERTa.trainModel()