from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Loading the pretrained BERT model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Tokenization and encoding
encodings = tokenizer(your_text_data, truncation=True, padding=True)

# Building PyTorch tensors for input
input_ids = torch.tensor(encodings['input_ids'])
attention_mask = torch.tensor(encodings['attention_mask'])
labels = torch.tensor(your_labels)

# Training loop
optimizer = AdamW(model.parameters(), lr=1e-5)  # You can adjust the learning rate
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
