from transformers import BertTokenizer
import torch

# Loading the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Text to tokenize
text = "This is an example sentence."

# Tokenizing and pad texting
tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# Extracting input tensors
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']
