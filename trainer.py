import math
import logging
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 1000
    learning_rate = 2e-3
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, device,ckptPath):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = TrainerConfig()
        self.ckptPath = ckptPath
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to("cuda")
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            # print('We are using the gpu now! device={}'.format(self.device))

        self.best_loss = 9e9

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.ckptPath)
        torch.save(raw_model.state_dict(), self.ckptPath)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            losses = []
            pbar = tqdm(enumerate(data), total=len(data)) if is_train else enumerate(data)
            for it, (x,_, y) in pbar:
                # place data on the correct device
                x = x.to(self.device) # input atom path representation
                y = y.to(self.device) # input elec tesor data
                x = x.unsqueeze(1)
                # y = y.unsqueeze(1)
                # print(x.shape, y.shape)
                y = y.repeat(1, 3, 3)
                # print("x.shape, y.shape",x.shape, y.shape)
                # forward the model
                with torch.set_grad_enabled(is_train):
                    loss, _, _ = model(x, y)
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.002, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        self.best_loss = float('inf') if self.best_loss is None else self.best_loss
        self.tokens = 0 # counter used for learning rate decay
        
        test_loss = 999999
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if (epoch + 1) % 1 == 0:
                if self.test_dataset is not None:
                    test_loss = run_epoch('test')
                # supports early stopping based on the test loss, or just save always if no test set is provided
                good_model = self.test_dataset is None or test_loss < self.best_loss
                if self.ckptPath is not None and good_model:
                    self.best_loss = test_loss
                    self.save_checkpoint()
