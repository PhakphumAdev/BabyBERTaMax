from transformers import BertTokenizer
from datasets import load_dataset
from params import params
from datasets import Dataset, DatasetDict
from typing import List, Dict, Any, Tuple
from pathlib import Path
from typing import Tuple, List, Dict
from itertools import islice

class babyDataset:
    def __init__(self, filepath, tokenizer):
        self.filepath = filepath
        self.tokenizer = tokenizer
        sentences = self.load_sentences_from_file(filepath,
                                            include_punctuation=params.include_punctuation,
                                            allow_discard=True)
        data_in_dict = {'text': self.make_sequences(sentences, params.num_sentences_per_input)}
        datasets = DatasetDict({'train': Dataset.from_dict(data_in_dict)})
        # use babyberta tokenizer to tokenize the dataset
        text_column_name = "text"
        self.tokenized_dataset =  datasets.map(
                        self.tokenize_function,
                        batched=True,
                        num_proc=4,
                        remove_columns=[text_column_name],
                        load_from_cache_file=True,
                    )
        
    #helper function from babyberta
    def load_sentences_from_file(file_path: Path,
                             include_punctuation: bool = True,
                             allow_discard: bool = False,
                             ) -> List[str]:
        """
        load sentences for language modeling from text file
        """

        print(f'Loading {file_path}', flush=True)

        res = []
        num_too_small = 0
        with file_path.open('r') as line_by_line_file:

            for sentence in line_by_line_file.readlines():

                if not sentence:  # during probing, parsing logic above may produce empty sentences
                    continue

                sentence = sentence.rstrip('\n')

                # check  length
                if sentence.count(' ') < params.Data.min_sentence_length - 1 and allow_discard:
                    num_too_small += 1
                    continue

                if not include_punctuation:
                    sentence = sentence.rstrip('.')
                    sentence = sentence.rstrip('!')
                    sentence = sentence.rstrip('?')

                res.append(sentence)

        if num_too_small:
            print(f'WARNING: Skipped {num_too_small:,} sentences which are shorter than {params.Data.min_sentence_length}.')

        return res
    
    #helper function from babyberta
    def make_sequences(sentences: List[str],
                   num_sentences_per_input: int,
                   ) -> List[str]:

        gen = (bs for bs in sentences)

        # combine multiple sentences into 1 sequence
        res = []
        while True:
            sentences_in_sequence: List[str] = list(islice(gen, 0, num_sentences_per_input))
            if not sentences_in_sequence:
                break
            sequence = ' '.join(sentences_in_sequence)
            res.append(sequence)

        print(f'Num total sequences={len(res):,}', flush=True)
        return res


    def tokenize_function(self,examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return self.tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=params.max_input_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )
