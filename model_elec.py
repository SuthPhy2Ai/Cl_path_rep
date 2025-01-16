import math
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

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


class SeqNetConfig:
    """Configuration for base PointNet model."""

    def __init__(self, embeddingSize, numberofPoints, **kwargs):
        self.embeddingSize = embeddingSize
        self.numberofPoints = numberofPoints  # Number of points
        for k, v in kwargs.items():
            setattr(self, k, v)


class SeqEmbedding(nn.Module):
    """Sequence embedding module."""

    def __init__(self, config):
        super(SeqEmbedding, self).__init__()
        self.fc = nn.Linear(9, 32)
        self.maxpool = nn.MaxPool1d(kernel_size=9, stride=1)
        self.min, self.max, self.steps = -20.0, 6.0, 9
        self.gamma = (self.max - self.min) / (self.steps - 1)
        self.register_buffer("filters", torch.linspace(self.min, self.max, self.steps))

        # Convolutional layers
        self.fc1 = nn.Conv1d(32, 8, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.fc2 = nn.Conv1d(8, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Conv1d(64, config.n_embd, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(config.n_embd)
        self.fc5 = nn.Linear(config.n_embd, config.n_embd)

        # Residual connections
        self.fc_res1 = nn.Conv1d(32, 64, kernel_size=5)
        self.bn_res1 = nn.BatchNorm1d(64)
        self.fc_res2 = nn.Conv1d(64, config.n_embd, kernel_size=5)
        self.bn_res2 = nn.BatchNorm1d(config.n_embd)
        
        # Activation functions and dropout
        self.max_pool = nn.MaxPool1d(3)
        self.act_fun = nn.ReLU()
        self.act_tanh = nn.GELU()
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, seq):
        # print("seq",seq.shape)
        # seq = seq.unsqueeze(-1).expand(-1, -1, 9)
        seq = torch.exp(-((seq - self.filters) ** 2) / self.gamma ** 2)
        seq = self.act_fun(self.fc(seq)).transpose(1, 2)

        # Convolutional layers with residual connections
        seq_res = seq
        seq = self.act_fun(self.bn1(self.fc1(seq)))
        seq = self.act_fun(self.bn2(self.fc2(seq)))
        seq = self.act_fun(self.bn_res1(self.fc_res1(seq_res)) + seq)
        seq_res = seq
        seq = self.act_fun(self.bn3(self.fc3(seq)))
        seq = self.act_fun(self.bn_res2(self.fc_res2(seq_res)) + seq)
        
        # Pooling and final transformations
        seq = self.max_pool(seq).squeeze()
        seq = self.drop(seq)
        return self.act_tanh(self.fc5(seq))


class CLIP(nn.Module):
    """CLIP model with sequence embedding and GRU-based attention."""

    def __init__(self, config, SeqNetConfig=None):
        super().__init__()
        self.config = config
        self.SeqNetConfig = SeqNetConfig
        self.pointNet = None

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
        self.points_emb = SeqEmbedding(config)
        self.fc_project_formula = nn.Linear(2 * config.n_embd, config.n_embd)
        self.fc_project_points = nn.Linear(config.n_embd, config.n_embd)

        # Scaling parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('total_labels', torch.arange(30000))

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.ln_p = nn.LayerNorm(config.n_embd)

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

    def forward(self, idx, points):
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
        # formula_embedding_final = self.ln_f(formula_embedding_final)
        formula_embedding_final = formula_embedding_final / formula_embedding_final.norm(dim=-1, keepdim=True)  # stru part named formula
        
        
        
        
        
        
        
        
        
        points_embeddings_final = self.points_emb(points) 
        points_embeddings_final = points_embeddings_final / points_embeddings_final.norm(dim=-1, keepdim=True)  # elec part named points
        
        # print("points_embeddings_final.shape:", points_embeddings_final.shape)
        # print("formula_embedding_final.shape:", formula_embedding_final.shape)
        logit_scale = self.logit_scale.exp()
        logits_per_points = logit_scale * points_embeddings_final @ formula_embedding_final.t()  
        logits_per_formula = logits_per_points.t()

        labels = self.total_labels[:b]

        # print("logits_per_formula.shape:", logits_per_formula.shape)
        # print("labels.shape:", labels.shape)

        if points.shape[0] == b:
            loss = (F.cross_entropy(logits_per_points, labels) +
                    F.cross_entropy(logits_per_formula, labels)) / 2
        else:
            loss = 0.0





        return loss, logits_per_points, logits_per_formula

    
    
if __name__ == "__main__":
    config = CLIPConfig(block_size=10, n_embd=32)
    model = CLIP(config)
    dummy_input = torch.randn(2, 5, 8, 32)
    dummy_points = torch.randn(2, 9)
    output = model(dummy_input, dummy_points)
    print("Model output shape:", output)
    