import argparse
import os

import torch
from torch import nn

import random
import numpy as np
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from easydict import EasyDict as edict
from datasets import build_dataset
import copy
from warnings import filterwarnings
import pickle
import yaml
from train_exp import train, train_0step
from ViT_exp import ViT_clf
from utils import get_world_size, set_random_seed, get_data

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-classes', type=int, default=50)
    parser.add_argument('--incremental-classes', type=int, default=5)
    parser.add_argument('--gpu',type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mix-up-first',type=str, default='True')
    parser.add_argument('--mix-up-increment',type=str, default='False')
    arg_input = parser.parse_args()

    initial_classes = arg_input.initial_classes
    incremental_classes = arg_input.incremental_classes
    gpu_num = arg_input.gpu
    seed = arg_input.seed
    mix_up_first = True if arg_input.mix_up_first=='True' else False
    mix_up_increment = True if arg_input.mix_up_increment=='True' else False

    f = open('./args/args.pkl','rb')
    args = pickle.load(f)
    f1 = open('./args/cifar_dytox.yaml', 'r', encoding='utf-8')
    args1 = yaml.safe_load(f1)
    f2 = open('./args/cifar100_order1.yaml', 'r', encoding='utf-8')
    args2 = yaml.safe_load(f2)

    args.update(args1)
    args.update(args2)
    args = edict(args)

    args.data_path = './dataset'
    args.data_set = 'CIFAR'
    args.initial_increment = initial_classes
    args.increment = incremental_classes
    total_step = (100-initial_classes)//incremental_classes
    total_step = total_step if initial_classes==0 else total_step+1
    args.log_category = f'{initial_classes}-{total_step}steps'
    args.output_basedir = ''
    args.distributed=False


    filterwarnings('ignore')
    set_random_seed(seed)
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'

    scenario_train, args.nb_classes = build_dataset(is_train=True, args=args)
    scenario_val, _ = build_dataset(is_train=False, args=args)

    lr = args.lr
    linear_scaled_lr = lr * args.batch_size * get_world_size() / 512.0
    args.lr = linear_scaled_lr


    criterion = nn.CrossEntropyLoss()

    for task_id in range(0,total_step):

        set_random_seed(seed)
        first_step_classes = incremental_classes if initial_classes == 0 else initial_classes
        model = ViT_clf(num_classes=first_step_classes, img_size=32, patch_size=4, num_patches=64, 
                    in_chans=3, embed_dim=384, depth=6, num_heads=args.num_heads, mlp_ratio=4.0, 
                    qkv_bias=False, qk_scale=False, drop_rate=0., attn_drop_rate=0.0, drop_path=args.drop_path, norm_layer=nn.LayerNorm, attention_type='GPSA')
        model.to(device)

        if task_id > 0:
            tokens = []
            task = 0
            for task in range(task_id-1):
                model.classifier_expand(incremental_classes)
                tokens.append(torch.load(f'./weights/state_{task}_{args.log_category}.pt', map_location=device)['task_tokens'])

            best_state = torch.load(f'./weights/state_{task_id-1}_{args.log_category}.pt', map_location=device)
            tokens.append(best_state['task_tokens'])
            if task_id > 1:
                model.set_teacher_task_token(nn.Parameter(torch.zeros(1, 1, 384)))
            model.load_state_dict(best_state)
            teacher_model = copy.deepcopy(model)
            model.classifier_expand(incremental_classes)
        
        
        val_loader_list = []
        for num in range(task_id):
            _, val_loader, _ = get_data(num, args, scenario_train, scenario_val)
            val_loader_list.append(val_loader)
        train_loader, val_loader, val_loader_entire = get_data(task_id, args, scenario_train, scenario_val)
        val_loader_list.append(val_loader)

        optimizer = create_optimizer(args, model)
        lr_scheduler, _ = create_scheduler(args, optimizer) 

        if task_id > 0:
            for p in teacher_model.parameters():
                p.requires_grad = False

            factor = 0.01
            if incremental_classes==10:
                epochs = 400
            elif incremental_classes==5:
                epochs = 300
            else:
                epochs = 500
            best_state, best_acc_val = train(model, teacher_model, train_loader, val_loader_entire, nn.ParameterList(tokens), task_id, initial_classes, incremental_classes,
                                            factor, criterion, optimizer, lr_scheduler, epochs, device, val_loader_list=val_loader_list, save_best=True, verbose=True,
                                            mixup=mix_up_increment, alpha=0.8)
        else:
            epochs = 500
            best_state, best_acc_val = train_0step(model, train_loader, val_loader_entire, criterion, optimizer, lr_scheduler, epochs, device, save_best=True, verbose=True, mixup=mix_up_first, alpha=0.8)
        print('------------------'*3)
        print(f'task_id : {task_id}')
        print(f'best_val : {best_acc_val}')
        torch.save(best_state, f'./weights/state_{task_id}_{args.log_category}.pt')
