import sys
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import copy
from convit import PatchEmbed, Block, MHSA, ClassAttention
from GPSA import GPSA
from timm.models.layers import trunc_normal_
from classifier import Classifier
from typing import Union, List

class ViT_clf(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, num_patches, in_chans, embed_dim, 
                 depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path, norm_layer,
                 attention_type='MHSA'):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio, self.qkv_bias, self.qk_scale = mlp_ratio, qkv_bias, qk_scale
        self.drop_rate, self.attn_drop_rate, self.norm_layer = drop_rate, attn_drop_rate, norm_layer
        self.mode = 'normal'

        self.patch_embedding = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.task_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        if attention_type=='GPSA':
            blocks = [self.SA_Blocks(GPSA, layer_index) for layer_index in range(depth-1)]
        elif attention_type=='MHSA':
            blocks = [self.SA_Blocks(MHSA, layer_index) for layer_index in range(depth-1)]
        
        blocks.append(self.SA_Blocks(ClassAttention, depth-1))
        self.blocks = nn.ModuleList(blocks)
        self.clf = nn.ModuleList([Classifier(embed_dim, nb_total_classes=None, nb_base_classes=num_classes, increment=10, nb_tasks=10)])  # nn.Linear층

    def set_task_token(self, task_tokens):
        self.task_tokens = nn.Parameter(task_tokens)
    
    def set_teacher_task_token(self, teacher_task_tokens):
        self.teacher_task_tokens = teacher_task_tokens

    def set_mode(self, mode='normal'):
        if mode in ['normal', 'kd']:
            self.mode = mode
        else:
            raise ValueError('mode should be in [normal, kd]')
        
    # 확장
    def classifier_expand(self, nb_new_classes):
        device = self.clf[0].norm.weight.device
        new_clf = Classifier(self.embed_dim, nb_total_classes=None, nb_base_classes=nb_new_classes, increment=10, nb_tasks=10)
        new_clf = new_clf.to(device)
        self.clf = nn.ModuleList([*self.clf, new_clf])

    def feature_extract(self, x):
        x = self.patch_embedding(x)
        x = self.pos_drop(x + self.pos_embed)
        if self.mode == 'normal':
            task_tokens = self.task_tokens.expand(x.shape[0],-1,-1)  
            x = torch.cat((task_tokens, x), dim=1)
        elif self.mode == 'kd':
            teacher_task_tokens = self.teacher_task_tokens.expand(x.shape[0],-1,-1) 
            x = torch.cat((teacher_task_tokens, x), dim=1)
        for block in self.blocks:
            x,_,_ = block(x)
        return x    
    
    def classify(self, x):
        x = torch.concat([clf(x[:,0]) for clf in self.clf], dim=1)
        return x
    
    def classiy_task(self, x, task_id):
        x = self.clf[task_id](x[:,0])
        return x
    
    def forward(self, x, task_id=None):
        x_emb = self.feature_extract(x)
        if task_id is None:
            x = self.classify(x_emb)
        else:
            x = self.classiy_task(x_emb, task_id)
        return x #, x_emb
    
    def SA_Blocks(self, attention_type, layer_index):
        return Block(
        dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
        drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=self.dpr[layer_index], norm_layer=self.norm_layer,
        attention_type=attention_type
    )

def forward(x, y, model, teacher_model, criterion, task_tokens_list:Union[List, torch.Tensor], is_weight=False, mixup=False, alpha=1.0):
    model.set_mode('normal')
    if mixup==True:
        x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=alpha, device=x.device)
        x_emb = model.feature_extract(x_mix)
        y_pred = model.classify(x_emb)
        loss_new_task = mixup_criterion(criterion, y_pred, y_a, y_b, lam)
    else:
        x_emb = model.feature_extract(x)
        y_pred = model.classiy_task(x_emb, len(task_tokens_list))
        loss_new_task = criterion(y_pred, y.long())

    loss_old_task = 0
    
    if task_tokens_list is not None:
        if is_weight:
            weights = np.linspace(0.5, 1, len(task_tokens_list))
            weights = weights / weights.sum()
        else:
            weights = [1] * len(task_tokens_list)
        for weight, task_tokens in zip(weights, task_tokens_list):
            model.set_teacher_task_token(task_tokens)
            teacher_model.set_task_token(task_tokens)
            
            model.set_mode('kd')
            xx_feature_old = model.feature_extract(x)
            xx_feaure_teacher = teacher_model.feature_extract(x)
            loss_each_old_task = torch.dist(xx_feature_old, xx_feaure_teacher, p=2)
            loss_old_task += weight*loss_each_old_task
    
    model.set_mode('normal')
    return y_pred, loss_new_task, loss_old_task

def inference(model, loader, task_id, task_token, device):
    model = copy.deepcopy(model)
    model.eval()
    model.to(device)
    model.set_task_token(task_token.to(device))
    logits = []
    targets = []
    for x, y,_ in loader:
        x = x.to(device)
        with torch.no_grad():
            logit = model(x, task_id)
        logits.append(logit)
        targets.append(y)
    logits = torch.cat(logits, dim=0)
    logits = logits.detach().cpu().numpy()
    targets = torch.cat(targets, dim=0)
    targets = targets.detach().cpu().numpy()
    return logits, targets

def mixup_data(x, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x.to(device), y_a.to(device), y_b.to(device), lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)