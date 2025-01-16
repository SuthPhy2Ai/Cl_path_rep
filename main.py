import os
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from ase.db import connect
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import groupby
import random
import logging
from src import data_proc
import multiprocessing as mp
from trainer import Trainer
from torch.utils.data import DataLoader, random_split
from model_elec import CLIP, CLIPConfig, SeqNetConfig
def set_seed(seed):
    random.seed(seed)       
    np.random.seed(seed)    
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) 
set_seed(42)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class CrystalDataset(Dataset):
    def __init__(self, db_path):
        self.db = connect(db_path)
        self.entries = list(self.db.select())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        tmp = self.entries[idx]
        atoms = tmp.toatoms()
        try:
            atom_feature = data_proc.get_crystal_path_muhead(ase_obj=True, stru=atoms, num_heads=4)
            target = torch.tensor(tmp.data['dielectric'], dtype=torch.float32)
            atom_feature = torch.tensor(atom_feature, dtype=torch.float32)
            return atom_feature, target
        except ValueError as e:
            print(f"Skipping structure {idx} due to error: {e}")
            return None 


def collate_fn(batch):
    """
    Custom collate function that handles skipping None samples and flattens the num_heads dimension.
    """
    batch = [b for b in batch if b is not None] 
    if len(batch) == 0:
        return None 
    
    atom_features, targets = zip(*batch)
    num_heads = atom_features[0].shape[0]
    max_atoms = max(feat.shape[1] for feat in atom_features)
    batch_size = len(atom_features)
    embed_dim = atom_features[0].shape[2]

    # Flatten the num_heads dimension
    padded_features = torch.zeros((batch_size * num_heads, max_atoms, embed_dim), dtype=torch.float32)
    attention_masks = torch.zeros((batch_size * num_heads, max_atoms), dtype=torch.float32)
    flattened_targets = torch.zeros((batch_size * num_heads, *targets[0].shape), dtype=torch.float32)

    for i, (feat, target) in enumerate(zip(atom_features, targets)):
        num_atoms = feat.shape[1]
        for head in range(num_heads):
            idx = i * num_heads + head
            padded_features[idx, :num_atoms, :] = feat[head]
            attention_masks[idx, :num_atoms] = 1
            flattened_targets[idx] = target

    return padded_features, attention_masks, flattened_targets

device = 'gpu'
numEpochs = 500
embeddingSize = 384  # emb design
batchSize = 128
blockSize = 31
num_workers = 4
dataInfo = 'Path_GRU'
addr = './Savemodels/'
fName = '{}.txt'.format(dataInfo)
ckptPath = '{}/{}.pt'.format(addr, fName.split('.txt')[0])
if not os.path.exists(addr):
    os.makedirs(addr)




dataset = CrystalDataset("/data/home/hzw1010/suth/elec_gw/dbs/clean.db")

train_size = int(0.75 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batchSize, pin_memory=False,shuffle=True, collate_fn=collate_fn, drop_last=False)
val_dataloader   = DataLoader(val_dataset, batch_size=batchSize,pin_memory=False,shuffle=False, collate_fn=collate_fn, drop_last=False)
test_dataloader  = DataLoader(test_dataset,batch_size=batchSize,pin_memory=False,shuffle=False, collate_fn=collate_fn, drop_last=False)

config = CLIPConfig(block_size=blockSize, n_embd=embeddingSize)
model = CLIP(config)


try:
    trainer = Trainer(model, train_dataloader, val_dataloader, device,ckptPath)
    trainer.train()
except:
    import traceback
    traceback.print_exc()
