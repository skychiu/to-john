MODEL = 'google/flan-t5-large'
BATCH_SIZE = 2
DEVICES = 1

import os
import sys

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

torch.set_float32_matmul_precision('medium')
tqdm.pandas()

df = pd.read_csv('data.csv')

tokenizer = T5Tokenizer.from_pretrained(MODEL)


def tokenize_function(row):
    text = "summarize: " + row['text']
    summary = row['headline']

    model_inputs = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(summary, max_length=128, truncation=True, padding='max_length', return_tensors='pt')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


df['tokenized'] = df.progress_apply(tokenize_function, axis=1)


class WikiHowDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].tokenized
        input_ids = item['input_ids'].squeeze()
        attention_mask = item['attention_mask'].squeeze()
        labels = item['labels'].squeeze()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


train_dataset = WikiHowDataset(df[~df.test])
val_dataset = WikiHowDataset(df[df.test])

num_cpus = os.cpu_count()


class T5FineTuner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=num_cpus, )

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_cpus, )

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)


summarization_model = T5FineTuner()


def main():
    global DEVICES
    if len(sys.argv) == 2:
        try:
            DEVICES = int(sys.argv[1])
        except:
            pass
    print('devices:', DEVICES)
    trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=DEVICES, )
    trainer.fit(summarization_model)


if __name__ == "__main__":
    main()
