from torch.utils.data import Dataset
import os
import pandas as pd

class TranslatorDataset(Dataset):
  def __init__(self, tokenizer, data_dir, type_path,  max_len=512):
    self.path = os.path.join(data_dir, type_path + '.csv')

    self.data_column = ["text"]
    self.class_column = ['target']
    self.data = pd.read_csv(self.path)
    self.max_len = max_len
    self.tokenizer = tokenizer
    self.inputs = []
    self.targets = []
    self._build()
  def __len__(self):
    return len(self.inputs)
  
  def __getitem__(self, index):
    source_ids = self.inputs[index]["input_ids"].squeeze()
    target_ids = self.targets[index]["input_ids"].squeeze()

    src_mask    = self.inputs[index]["attention_mask"].squeeze()  
    target_mask = self.targets[index]["attention_mask"].squeeze()  

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  
  def _build(self):
    for idx in range(self.data.shape[0]):
      input_ =  self.data.loc[idx][self.data_column]
      target =  self.data.loc[idx, self.class_column].values
      input_ = str(input_) + ' </s>'
      target = str(target) + ' </s>'
       # tokenize inputs
      tokenized_inputs = self.tokenizer.batch_encode_plus(
          [input_], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
      tokenized_targets = self.tokenizer.batch_encode_plus(
          [target], max_length= self.max_len, pad_to_max_length=True, return_tensors="pt"
      )
      self.inputs.append(tokenized_inputs)
      self.targets.append(tokenized_targets)