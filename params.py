#params from BabyBERTa
class params:
    # Data related settings
    sample_with_replacement = False
    training_order = 'original'
    consecutive_masking = False
    num_sentences_per_input = 1
    include_punctuation = True
    allow_truncated_sentences = False
    num_mask_patterns = 10
    mask_pattern_size = 2
    probabilistic_masking = True
    mask_probability = 0.15
    leave_unmasked_prob_start = 0.0
    leave_unmasked_prob = 0.0
    random_token_prob = 0.1
    #corpora = ('aochildes',)
    #we will apply curriculum learning here by training from child-directed speech first then complex and non-dialogue text
    corpora = ('childes','bnc_spoken','switchboard','open_subtitles','gutenberg','simple_wiki',)
    tokenizer = 'babyberta'
    add_prefix_space = True
    max_input_length = 128

    # Training specific settings
    batch_size = 16
    lr = 1e-4
    num_epochs = 1
    num_warmup_steps = 24000
    weight_decay = 0.0

    # Model configuration
    load_from_checkpoint = 'none'
    hidden_size = 256
    num_layers = 8
    num_attention_heads = 8
    intermediate_size = 1024
    initializer_range = 0.02
    layer_norm_eps = 1e-5

    #from config
    min_sentence_length = 3

