![](distilbert.png)
# coronaBERT: DistilBERT-Based Model for COVID-19 Policy classification
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Joesh1/coronaBERT)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)


## Overview
- This repository contains code, training scripts, and documentation for a `fine-tuned DistilBERT` model for national policy related to COVID-19 classification tasks. The model is designed to classify text descriptions of COVID policies into predefined categories (Policy Type). The project includes data preprocessing, model training, evaluation, and quantization for efficient inference.

## Feature
- Fine-tuned on custom labeled data for sequence classification
- Supports dynamic quantization for reduced model size and faster inference
- End-to-end pipeline: data preparation, training, evaluation, and model export
- Ready-to-use with Hugging Face Transformers

## Model detail
- Base model: `distilbert-base-uncased`
- Data: [CoronaNet Research Project](https://coronanet-project.org/) 
- Task: Sequence Classification (e.g., predicting type)
- Framework: PyTorch, Transformers
- Quantization: Optional INT8 dynamic quantization for deployment

## How to use it

```
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# download and load model
model = AutoModelForSequenceClassification.from_pretrained(
    "Joesh1/coronaBERT", device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("Joesh1/coronaBERT")

# Inference
classifier = pipeline("text-classification", model="Joesh1/coronaBERT")
classifier("The government initiated COVID-19 vaccine distribution as of..")

# output
#[{'label': 'COVID-19 Vaccines', 'score': 0.9725420475006104}]
```

## Try the model on HuggingFace
- [coronaBERT](https://huggingface.co/Joesh1/coronaBERT)
