import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from ase.db import connect
from src import data_proc

# 设置设备
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义 Schmidt 正交化函数（用于处理目标值）
def schmidt_orthogonalization(vectors):
    vectors = vectors.reshape(3, 3)
    diagonal_elements = np.diagonal(vectors)
    ele = np.mean(diagonal_elements)
    return ele

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 自定义数据集类
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
            atom_feature = data_proc.get_crystal_path_muhead(ase_obj=True, stru=atoms, num_heads=1)
            target = torch.tensor(tmp.data['dielectric'], dtype=torch.float32)  # 保留目标属性
            atom_feature = torch.tensor(atom_feature, dtype=torch.float32)
            return atom_feature, target  # 返回晶体特征和目标属性
        except ValueError as e:
            print(f"Skipping structure {idx} due to error: {e}")
            return None

# 自定义 collate 函数
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    atom_features, targets = zip(*batch)
    num_heads = atom_features[0].shape[0]
    max_atoms = max(feat.shape[1] for feat in atom_features)
    batch_size = len(atom_features)
    embed_dim = atom_features[0].shape[2]

    # 填充特征并展平 num_heads 维度
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

# 定义 GRU 编码器
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)  # h_n: (num_layers, batch_size, hidden_dim)
        h_n = h_n[-1]  # 取最后一层的隐藏状态
        output = self.fc(h_n)
        return output

class ContrastiveModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super(ContrastiveModel, self).__init__()
        self.encoder = GRUEncoder(feature_dim, hidden_dim, output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x1, x2=None):
        if x2 is None:
            return self.encoder(x1)
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return z1, z2
# 加载预训练模型
feature_dim = 384
hidden_dim = 256
output_dim = 128
model = ContrastiveModel(feature_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load("best_path_feature.pth"))
model.eval()

# 加载数据集
dataset = CrystalDataset("/data/home/hzw1010/suth/elec_gw/dbs/clean.db")
dataloader = DataLoader(dataset, batch_size=64, num_workers=2, pin_memory=False,
                        shuffle=False, collate_fn=collate_fn, drop_last=False)

# 提取特征
def extract_features(model, dataloader, device):
    features = []
    targets = []
    with torch.no_grad():
        for batch_features, batch_masks, batch_targets in tqdm(dataloader, desc="Extracting Features"):
            if batch_features is None:
                continue
            batch_features = batch_features.to(device)
            z = model(batch_features)  # 提取特征
            features.append(z.cpu())
            targets.append(batch_targets.cpu())
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)

# 处理目标值
def process_targets(targets):
    processed_targets = []
    for target in targets:
        elec = schmidt_orthogonalization(target.numpy())
        processed_targets.append(elec)
    return torch.tensor(processed_targets, dtype=torch.float32)

# 提取特征并处理目标值
features, targets = extract_features(model, dataloader, device)
processed_targets = process_targets(targets)
import pandas as pd
from autogluon.tabular import TabularPredictor
dataset_size = len(features)
train_size = int(0.8 * dataset_size)  
train_data = pd.DataFrame(features[:train_size].numpy())
train_labels = pd.Series(processed_targets[:train_size].numpy())
val_data = pd.DataFrame(features[train_size:].numpy())
val_labels = pd.Series(processed_targets[train_size:].numpy())
predictor = TabularPredictor(label="target", eval_metric="mean_absolute_error")
train_data['target'] = train_labels