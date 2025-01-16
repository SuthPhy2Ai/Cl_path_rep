import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CLIPConfig:
    """Configuration for CLIP model."""
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class CLIP(nn.Module):
    """CLIP model with sequence embedding and GRU-based attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        embeddingSize = config.n_embd
        self.block_size = config.block_size
        # Linear layers for attention scoring
        self.fc4_ = nn.Linear(4, 1)
        self.fc24_ = nn.Linear(24, 1)
        
        # Learnable weights for attention
        self.watt1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.watt2 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.watt3 = nn.Parameter(torch.ones(1), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.ln_gru1 = nn.LayerNorm(config.n_embd)
        self.dropkey1 = nn.Dropout(config.attn_pdrop)
        
        # Bi-directional GRU layers
        self.gru1 = nn.GRU(bidirectional=True, hidden_size=embeddingSize, batch_first=True, input_size=embeddingSize)
        self.att_score = nn.Linear(2 * embeddingSize, 1, bias=False)
        self.fc_proj = nn.Linear(2 * embeddingSize, 2 * embeddingSize)
        
        self.ln_gru2 = nn.LayerNorm(2 * config.n_embd)
        self.gru2 = nn.GRU(bidirectional=True, hidden_size=embeddingSize, batch_first=True, input_size=2 * embeddingSize)
        self.att_score2 = nn.Linear(2 * embeddingSize, 1, bias=False)
        
        # Embeddings and normalization
        self.penalty_labels = torch.eye(config.block_size)
        self.fc_project_formula = nn.Linear(2 * config.n_embd, config.n_embd)

        # Scaling parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('total_labels', torch.arange(30000))

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)
        logger.info("Number of parameters: %e", sum(p.numel() for p in self.parameters()))
        
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.GRU)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.startswith('gru1.'):
                    decay.add(fpn)
                elif pn.startswith('gru2.'):
                    decay.add(fpn)
                elif pn.startswith('layer.'):
                    decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('logit_scale')
        decay.update(['watt1', 'watt2', 'watt3'])
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx):
        b, l1, l2, e = idx.size()  
        formula_mask = idx[:, :, 0, 0] == -1.0
        idx = idx.view(-1, l2, e)  
        the_rs      = idx[:, :, -24:] 
        the_windows = idx[:, :, -28:-24] 
        pos_of_wins_att = self.sigmoid(self.fc4_(the_windows))
        pos_of_rs_att = self.sigmoid(self.fc24_(the_rs))
        x, _ = self.gru1(self.ln_gru1(idx))
        mask = idx[:, :, 0] == -1.0
        att_score = self.watt1 * self.att_score(x)    +  self.watt2* pos_of_wins_att  +   2 * self.watt3* pos_of_rs_att
        mask = 1.0 - mask.float()
        # mask = self.dropkey1(mask)
        att_score = torch.where(mask==0.0, torch.ones_like(mask) * -1e10, att_score.squeeze(-1))
        att_weights = torch.softmax(att_score, dim=1)
        x = torch.sum(att_weights.unsqueeze(-1) * x, dim=1)
        x = x.view(b, l1, 2 * e)

        x = self.fc_proj(x)
        x, _ = self.gru2(self.ln_gru2(x))
        mask = formula_mask
        mask = 1.0 - mask.float()
        mask = self.dropkey1(mask)
        att_score = self.att_score2(x)
        att_score = torch.where(mask==0.0, torch.ones_like(mask) * -1e10, att_score.squeeze(-1))

        att_weights = torch.softmax(att_score, dim=1)
        x = torch.sum(att_weights.unsqueeze(-1) * x, dim=1)


        formula_embedding_final = self.fc_project_formula(x)
        formula_embedding_final = formula_embedding_final / formula_embedding_final.norm(dim=-1, keepdim=True)  # stru part named formula
        
        # Self-similarity calculation
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * formula_embedding_final @ formula_embedding_final.t()  

        # Generate labels for self-similarity
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)  # Labels are [0, 1, ..., batch_size-1]

        # Use generated labels for loss calculation
        loss = F.cross_entropy(logits, labels)

        return loss, logits


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (torch.Tensor): Input data of shape (num_samples, seq_len, num_atoms, embedding_dim).
            labels (torch.Tensor): Labels for each sample of shape (num_samples,).
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    """
    Collate function to group samples with the same label together.
    Args:
        batch: List of tuples (data, label).
    Returns:
        batched_data: Tensor of shape (batch_size, seq_len, num_atoms, embedding_dim).
    """
    # Sort batch by labels
    batch = sorted(batch, key=lambda x: x[1])
    data, _ = zip(*batch)
    batched_data = torch.stack(data, dim=0)
    return batched_data


# Example data
data = torch.randn(100, 5, 8, 32)  # 100 samples, each with shape (5, 8, 32)
labels = torch.randint(0, 10, (100,))  # 100 labels, ranging from 0 to 9

# Create dataset and dataloader
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Initialize model and optimizer
config = CLIPConfig(block_size=10, n_embd=32)
model = CLIP(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):  # 10 epochs
    for batch_data in dataloader:
        optimizer.zero_grad()
        loss, _ = model(batch_data)  # No need to pass labels
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")