import torch
from torch.utils.data import Dataset, DataLoader
from ase.db import connect
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from tqdm.auto import tqdm
import time
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from src import data_proc



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
            target = float(tmp.mpid.split("-")[-1])   # run self find do not need target
            atom_feature = torch.tensor(atom_feature, dtype=torch.float32)
            return atom_feature, target 
        except ValueError as e:
            print(f"Skipping structure {idx} due to error: {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    atom_features, targets = zip(*batch)
    num_heads = atom_features[0].shape[0]
    max_atoms = max(feat.shape[1] for feat in atom_features)
    batch_size = len(atom_features)
    embed_dim = atom_features[0].shape[2]
    padded_features = torch.zeros((batch_size * num_heads, max_atoms, embed_dim), dtype=torch.float32)
    attention_masks = torch.zeros((batch_size * num_heads, max_atoms), dtype=torch.float32)
    flattened_targets = torch.zeros((batch_size * num_heads, *torch.tensor(targets).shape), dtype=torch.float32)

    for i, (feat, target) in enumerate(zip(atom_features, targets)):
        num_atoms = feat.shape[1]
        for head in range(num_heads):
            idx = i * num_heads + head
            padded_features[idx, :num_atoms, :] = feat[head]
            attention_masks[idx, :num_atoms] = 1
            flattened_targets[idx] = target

    return padded_features, attention_masks, flattened_targets


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, activation_wins='sigmoid', activation_rs='tanh'):
        super(GRUEncoder, self).__init__()
        # 第一层GRU
        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # 注意力分数计算层
        self.att_score = nn.Linear(2 * hidden_dim, 1)  # 双向GRU的输出维度是2 * hidden_dim
        
        # 其他注意力分数的计算层
        self.fc4_ = nn.Linear(4, 1)  # 用于计算pos_of_wins_att
        self.fc24_ = nn.Linear(24, 1)  # 用于计算pos_of_rs_att
        
        # 可学习的权重参数
        self.watt1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.watt2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.watt3 = nn.Parameter(torch.ones(1), requires_grad=True)
        
        # 投影层
        self.fc_proj = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        
        # 第二层GRU
        self.gru2 = nn.GRU(2 * hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.att_score2 = nn.Linear(2 * hidden_dim, 1)  # 双向GRU的输出维度是2 * hidden_dim
        
        # 最终输出层
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        
        # 激活函数
        self.activation_wins = activation_wins
        self.activation_rs = activation_rs
        
        # 定义激活函数
        self.act_fn_wins = self._get_activation_fn(activation_wins)
        self.act_fn_rs = self._get_activation_fn(activation_rs)
        self.dropatt = nn.Dropout(0.1)

    def _get_activation_fn(self, activation):
        if activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x, attention_mask):
        # 提取输入特征中的特定部分
        the_windows = x[:, :, -28:-24]  # 假设x的最后一个维度是28
        the_rs = x[:, :, -24:]  # 假设x的最后一个维度是24
        
        # 计算其他注意力分数
        pos_of_wins_att = self.act_fn_wins(self.fc4_(the_windows))  # (batch_size, seq_len, 1)
        pos_of_rs_att = self.act_fn_rs(self.fc24_(the_rs))  # (batch_size, seq_len, 1)
        
        # 第一层GRU
        x, _ = self.gru1(x)  # x: (batch_size, seq_len, 2 * hidden_dim)
        
        # 计算注意力分数
        att_score = self.watt1 * self.att_score(x) + self.watt2 * pos_of_wins_att + 2 * self.watt3 * pos_of_rs_att
        att_score = att_score.squeeze(-1)  # (batch_size, seq_len)
        
        # 应用掩码
        att_score = att_score.masked_fill(attention_mask == 0, -1e10)  # 将填充部分的值设为负无穷
        att_weights = F.softmax(att_score, dim=1)  # 对序列维度进行softmax
        
        # 使用注意力权重对GRU的输出进行加权求和
        attended_output1 = torch.sum(att_weights.unsqueeze(-1) * x, dim=1)  # (batch_size, 2 * hidden_dim)
        
        # 投影层
        projected_output = self.fc_proj(attended_output1)  # (batch_size, 2 * hidden_dim)
        
        # 第二层GRU
        x, _ = self.gru2(projected_output.unsqueeze(1))  # x: (batch_size, seq_len, 2 * hidden_dim)
        
        # 第二层注意力
        att_score2 = self.att_score2(x)  # (batch_size, seq_len, 1)
        att_score2 = att_score2.squeeze(-1)  # (batch_size, seq_len)
        
        # 应用掩码
        att_score2 = att_score2.masked_fill(attention_mask == 0, -1e10)  # 将填充部分的值设为负无穷
        att_weights2 = F.softmax(att_score2, dim=1)  # 对序列维度进行softmax
        
        # 使用注意力权重对GRU的输出进行加权求和
        attended_output2 = torch.sum(att_weights2.unsqueeze(-1) * x, dim=1)  # (batch_size, 2 * hidden_dim)
        
        # 最终输出
        output = self.fc(attended_output2)  # (batch_size, output_dim)
        return output.squeeze(dim=1)

class ContrastiveModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super(ContrastiveModel, self).__init__()
        self.encoder = GRUEncoder(feature_dim, hidden_dim, output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x1, x2,batch_masks):
        z1 = self.encoder(x1,batch_masks)
        z2 = self.encoder(x2,batch_masks)
        # print(f"z1 shape: {z1.shape}, z2 shape: {z2.shape}")
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * z1 @ z2.t()
        return logits

def contrastive_loss(logits):
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return loss

# 测试函数
def test(model, dataloader, device, num_heads=4):
    model.eval()  # 设置为评估模式
    model.to(device)
    total_correct_top1 = 0
    total_correct_top3 = 0
    total_correct_top5 = 0
    total_correct_top10 = 0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算
        for batch_features, batch_masks, batch_targets in tqdm(dataloader, desc="Testing", unit="batch"):
            if batch_features is None:
                continue

            # 将数据移动到指定设备
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            batch_masks  = batch_masks.to(device)
            # print(batch_targets)
            # 提取高级特征
            z = model.encoder(batch_features,batch_masks)  # z 是高级特征表示

            # 归一化特征
            z = F.normalize(z, p=2, dim=-1)

            # 计算相似度矩阵
            similarity_matrix = z @ z.t()  # 相似度矩阵 (batch_size * num_heads, batch_size * num_heads)

            # 获取每个样本的 top-k 最相似样本（排除自身）
            _, indices = torch.topk(similarity_matrix, k=11, dim=1)  # 取 top-11，排除自身
            # indices = indices[:, 1:]  # 去掉自身，剩下 top-10   !!!! 不去掉自己

            # 检查预测是否正确
            batch_targets_expanded = batch_targets.unsqueeze(1)  # (batch_size * num_heads, 1, target_dim)
            predicted_targets = batch_targets[indices]  # (batch_size * num_heads, 10, target_dim)
            correct = (predicted_targets == batch_targets_expanded).all(dim=-1)  # (batch_size * num_heads, 10)

            # 对 num_heads 进行去重
            correct = correct.view(-1, num_heads, correct.size(1))  # (batch_size, num_heads, 10)
            correct = correct.any(dim=1)  # (batch_size, 10)

            # 统计 top-k 正确匹配的数量
            total_correct_top1 += correct[:, 0].sum().item()  # Top-1
            total_correct_top3 += correct[:, :3].any(dim=1).sum().item()  # Top-3
            total_correct_top5 += correct[:, :5].any(dim=1).sum().item()  # Top-5
            total_correct_top10 += correct[:, :10].any(dim=1).sum().item()  # Top-10
            total_samples += correct.size(0)

    # 计算准确率
    accuracy_top1 = total_correct_top1 / total_samples
    accuracy_top3 = total_correct_top3 / total_samples
    accuracy_top5 = total_correct_top5 / total_samples
    accuracy_top10 = total_correct_top10 / total_samples

    # 输出表格
    table = [
        ["Top-1", f"{accuracy_top1 * 100:.2f}%"],
        ["Top-3", f"{accuracy_top3 * 100:.2f}%"],
        ["Top-5", f"{accuracy_top5 * 100:.2f}%"],
        ["Top-10", f"{accuracy_top10 * 100:.2f}%"],
    ]
    print(tabulate(table, headers=["Metric", "Accuracy"], tablefmt="pretty"))

    return {
        "Top-1": accuracy_top1,
        "Top-3": accuracy_top3,
        "Top-5": accuracy_top5,
        "Top-10": accuracy_top10,
    }



device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Using device: {device}")
dataset = CrystalDataset("/data/home/hzw1010/suth/elec_gw/dbs/mpdata.db")


# import random
# from torch.utils.data import Subset
# total_size = len(dataset)
# subset_size = total_size // 10
# indices = random.sample(range(total_size), subset_size)
# train_subset = Subset(dataset, indices)
# dataset = train_subset

train_dataloader = DataLoader(dataset, batch_size=128, num_workers=2, pin_memory=False,
                              shuffle=True, collate_fn=collate_fn, drop_last=False)
test_dataloader = DataLoader(dataset, batch_size=256, num_workers=2, pin_memory=False,
                             shuffle=False, collate_fn=collate_fn, drop_last=False)

# 初始化模型和优化器
model = ContrastiveModel(feature_dim=384, hidden_dim=256, output_dim=128).to(device)  # 将模型移动到指定设备
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 早停逻辑
best_top1_accuracy = 0.0
epochs_without_improvement = 0
max_epochs_without_improvement = 3

# 训练循环
for epoch in range(100):  # 设置一个较大的 epoch 上限
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch")

    for batch_features, batch_masks, batch_targets in progress_bar:
        if batch_features is None:
            continue

        x1 = batch_features.to(device)
        x2 = batch_features.to(device)
        batch_targets = batch_targets.to(device)
        batch_masks = batch_masks.to(device)
        optimizer.zero_grad()

        logits = model(x1, x2, batch_masks)
        loss = contrastive_loss(logits)
        loss.backward()
        optimizer.step()

        # 更新 epoch 损失
        epoch_loss += loss.item()

        # 更新进度条描述，显示当前 batch 的损失值
        progress_bar.set_postfix({"Loss": loss.item()})

    # 打印每个 epoch 的平均损失
    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    test_results = test(model, test_dataloader, device, num_heads=4)
    current_top1_accuracy = test_results["Top-1"]
    if current_top1_accuracy > best_top1_accuracy:
        best_top1_accuracy = current_top1_accuracy
        epochs_without_improvement = 0
        torch.save(model.state_dict(), f"best_path_feature.pth")
        print(f"Model saved to 'best_model.pth with {best_top1_accuracy:.4f}'.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= max_epochs_without_improvement:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break