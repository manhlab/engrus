from pytorch_lightning import ModelCheckpoint
from pytorch_lightning import Trainer
from model_store.t5_model import T5Lightning
from translater import seed_all, TranslatorDataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
import wandb
checkpoint_callback = ModelCheckpoint(filepath='t5_segment')
trainer = Trainer(
    checkpoint_callback=checkpoint_callback,
    weights_save_path='t5_segment'
)
model = T5Lightning(args)
wandb.restore('model/t5model.pth')
model.load_from_checkpoint('model/t5model.pth')
seed_all(34)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model.model.test()
input_ = "implement t5 model english russian </s>"
input_ =  tokenizer.batch_encode_plus(
          [input_], max_length=512, pad_to_max_length=True, return_tensors="pt")
outs = model.model.generate(input_ids=input_['source_ids'].cuda(), \
                              attention_mask=input_['source_mask'].cuda(),\ 
                              max_length=2)
dec = [tokenizer.decode(ids) for ids in outs]  
print(dec)