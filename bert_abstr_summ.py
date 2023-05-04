#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:38:48 2023

@author: avi_patel
"""

import pandas as pd
import torch
from transformers import BertTokenizer, EncoderDecoderModel, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Set up parameters
max_input_length = 512  # Maximum length of input sequence
max_output_length = 256  # Maximum length of output sequence
batch_size = 8
learning_rate = 5e-5
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load comments from DataFrame
df = pd.read_csv('your_dataframe.csv')
comments = df['comment'].tolist()

# Set up tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
model.to(device)

# Set up dataset and dataloader
class CommentDataset(Dataset):
    def __init__(self, comments, tokenizer, max_length):
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, idx):
        comment = self.comments[idx]
        input_ids = self.tokenizer.encode(comment, add_special_tokens=True, max_length=self.max_length, truncation=True, padding='max_length')
        return {'input_ids': torch.tensor(input_ids)}

dataset = CommentDataset(comments, tokenizer, max_input_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Fine-tune model for abstractive summarization
model.train()
for epoch in range(num_epochs):
    for batch in tqdm(dataloader, desc=f"Training epoch {epoch+1}/{num_epochs}"):
        input_ids = batch['input_ids'].to(device)
        decoder_input_ids = input_ids[:, :max_output_length].clone().detach()
        decoder_input_ids[:, 1:] = -100
        labels = input_ids[:, 1:max_output_length].clone().detach()

        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Generate abstractive summaries for the input comments
model.eval()
summaries = []
with torch.no_grad():
    for comment in tqdm(comments, desc="Generating summaries"):
        input_ids = tokenizer.encode(comment, add_special_tokens=True, max_length=max_input_length, truncation=True, padding='max_length')
        decoder_input_ids = torch.ones((1, max_output_length), dtype=torch.long, device=device) * tokenizer.pad_token_id
        decoder_input_ids[:, 0] = tokenizer.cls_token_id

        for step in range(max_output_length - 1):
            outputs = model(input_ids=torch.tensor([input_ids], device=device), decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[0, step, :]
            next_token_id = torch.argmax(logits).item()
            decoder_input_ids[:, step+1] = next_token_id

            if next_token_id == tokenizer.sep_token_id:
                break

        summary = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
        summaries.append(summary)

# Filter summaries by length
filtered_sum
