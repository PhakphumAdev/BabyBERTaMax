# BoyBERTaMax

## Overview
This repository contains the source code for BoyBERTa, an optimized version of the BabyBERTa model developed to enhance grammar learning from child-directed speech. BoyBERTa utilizes a novel curriculum learning approach to efficiently train on a corpus of 10 million words, advancing from simpler to more complex linguistic tasks.

## Key Features
- **Curriculum Learning**: Trains progressively on a structured curriculum from basic to advanced grammar structures.
- **Performance Evaluation**: Utilizes the BLiMP benchmark to measure training effectiveness and compare against conventional training methods.
- **Comparison Benchmarks**: Includes comparative analysis with baseline BabyBERTa and RoBERTa models, demonstrating improvements and efficiencies.

## Repository Structure
- `dataset/`: Contains scripts and utilities to manage the training datasets.
- `saved_models/`: Our final models: curriculum learning and random order.
- `BabyLM_Evaluation.ipynb`: Our result on BabyLM evaluation pipeline (BLiMP tasks)
 
## Usage
To train our model use
```python
python main.py true # to train with curriculum learning
python main.py false # to train without curriculum learning (random order)
```
