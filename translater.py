import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics
import json
from model_store.t5_model import T5Lightning,LoggingCallback
from data_processing import data_processing
from dataclass import TranslatorDataset

def seed_all(seed):
  random.seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
def get_dataset(tokenizer, type_path, args):
      return TranslatorDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,  max_len=args.max_seq_length)
seed_all(34)
data_processing()

with open('config.json', 'r') as f:
    args_dict = json.load(f)
args = argparse.Namespace(**args_dict)
model = T5Lightning(args)
print("T5Model created\n===========\n")
tokenizer = T5Tokenizer.from_pretrained('t5-base')
dataset = TranslatorDataset(tokenizer, 'data', 'train', 512)
print(len(dataset))



checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)
train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)
print("Training Model")
trainer = pl.Trainer(**train_params)
trainer.fit(model)
print("Trained model success!")
trainer.save_checkpoint('model/t5model.pth')
wandb.save('model/t5model.pth')
print("save model to model/t5model.pth")
wandb.restore('model/t5model.pth')
model.load_from_checkpoint('model/t5model.pth')
print("load model from model/t5model.pth")
# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
loader = DataLoader(dataset, batch_size=32, shuffle=True)
it = iter(loader)
batch = next(it)
print(batch["source_ids"].shape)
outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                              attention_mask=batch['source_mask'].cuda(), 
                              max_length=2)

dec = [tokenizer.decode(ids) for ids in outs]
texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

for i in range(12):
    c = texts[i]
    lines = textwrap.wrap("text:\n%s\n" % c, width=100)
    print("\n".join(lines))
    print("\nActual sentiment: %s" % targets[i])
    print("predicted sentiment: %s" % dec[i])
    print("=====================================================================\n")



