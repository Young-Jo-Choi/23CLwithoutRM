import torch
import numpy as np
import copy
from ViT_exp import forward, inference, mixup_data, mixup_criterion
from sklearn.metrics import accuracy_score
from typing import List

def train_0step(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, epochs, device, task_num=0, 
                save_best=False, verbose=False, mixup=False, alpha=1.0):
    model.to(device)
    best_state = model.state_dict()
    best_val_acc = 0.0
    for epoch in range(epochs):
        loss_tr = 0
        correct_tr = 0
        count_tr = 0
        loss_val = 0
        correct_val = 0
        count_val = 0
        model.train()
        for images, targets, _ in train_loader:
            targets -= task_num*10
            targets = targets.to(device, non_blocking=True)
            if mixup == True:
                images, targets_a, targets_b,lam = mixup_data(images, targets, alpha=alpha, device=device)
                outputs = model(images, 0)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                images = images.to(device, non_blocking=True)
                outputs = model(images, 0)
                loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tr += loss.item()
            correct_tr += (outputs.argmax(1) == targets).sum().item()
            count_tr += len(targets)
        lr_scheduler.step(epoch)

        model.eval()
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device, non_blocking=True)
                targets -= task_num*10
                targets = targets.to(device, non_blocking=True)
                outputs = model(images, 0)
                loss = criterion(outputs, targets)
                loss_val += loss.item()
                correct_val += (outputs.argmax(1) == targets).sum().item()
                count_val += len(targets)
            if best_val_acc < correct_val/count_val:
                best_val_acc = correct_val/count_val
                best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}\nloss_tr: {loss_tr/count_tr:.6f} acc_tr: {correct_tr/count_tr:.4f}\nloss_val: {loss_val/count_val:.6f} acc_val: {correct_val/count_val:.4f}")
        if save_best==False:
            best_state = copy.deepcopy(model.state_dict())
    return best_state, best_val_acc

def train(model,teacher_model, train_loader, val_loader, task_tokens_old, task_id, initial_classes, incremental_classes,
          factor, criterion, optimizer, lr_scheduler, epochs, device, val_loader_list, verbose=False, save_best=False,
          mixup=False, alpha=1.0):
    best_acc_val = 0.0

    if initial_classes == 0:
        num_classes_per_task = [0] + list(range(incremental_classes, 101, incremental_classes))
    else:
        num_classes_per_task = [0] + list(range(initial_classes, 101, incremental_classes))
    for epoch in range(epochs):
        for images, targets, _ in train_loader:
            optimizer.zero_grad()
            # (x, y, model, teacher_model, criterion, task_tokens_list:Union[List, torch.Tensor], is_distill=True, is_weight=False
            _, loss_new_task, loss_old_task = forward(x=images.to(device), y=targets.to(device)-num_classes_per_task[task_id], 
                                                      model=model, 
                                                      teacher_model=teacher_model, 
                                                      criterion=criterion, 
                                                      task_tokens_list=task_tokens_old, 
                                                      is_weight=False, 
                                                      mixup=mixup, alpha=alpha)
            loss = loss_new_task + factor * loss_old_task
            loss.backward()
            optimizer.step()
        lr_scheduler.step(epoch)
        task_token_new = copy.deepcopy(model.task_tokens)
        model.eval()
        inner_acc_all = []
        with torch.no_grad():
            for old_task_id in range(len(task_tokens_old)):
                task_token_old = task_tokens_old[old_task_id]
                val_loader_each = val_loader_list[old_task_id]
                logits, targets = inference(model, val_loader, old_task_id, task_token_old, device)
                logits_each, targets_each = inference(model, val_loader_each, old_task_id, task_token_old, device)
                targets_each -= num_classes_per_task[old_task_id]
                inner_acc_each = accuracy_score(targets_each, logits_each.argmax(1))
                inner_acc_all.append(inner_acc_each)
                if old_task_id == 0:
                    logits_all = logits
                else:
                    logits_all = np.c_[logits_all, logits]
            logits,_ = inference(model, val_loader, old_task_id+1, task_token_new, device)
            logits_each, targets_each = inference(model, val_loader_list[-1], old_task_id+1, task_token_new, device)
            inner_acc_each = accuracy_score(targets_each - num_classes_per_task[old_task_id+1], logits_each.argmax(1))
            inner_acc_all.append(inner_acc_each)
            logits_all = np.c_[logits_all, logits]
        acc_val = accuracy_score(targets, logits_all.argmax(1))
        each_classes_num = np.array(num_classes_per_task)[1:] - np.array(num_classes_per_task)[:-1]
        inner_acc_val = (np.array(inner_acc_all) * each_classes_num[:len(inner_acc_all)]).sum() / each_classes_num[:len(inner_acc_all)].sum()
        if best_acc_val < inner_acc_val:
            best_acc_val = inner_acc_val
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}\nacc_val: {acc_val:.4f}")
            print(f"inner_acc_all: {inner_acc_val}")
            if initial_classes == 0:
                each_accuracy_print(targets, logits_all.argmax(1), [incremental_classes]*(old_task_id+2))
            else: 
                each_accuracy_print(targets, logits_all.argmax(1), [initial_classes]+[incremental_classes]*(old_task_id+1))
            for i, inner_acc in enumerate(inner_acc_all):
                print(f'task{i} inner_acc_val : {inner_acc:.4f}', end='\t')
            print()
    if save_best==False:
        best_state = copy.deepcopy(model.state_dict())
        
    return best_state, best_acc_val

def each_accuracy_print(targets, predictions, increments_classes:List[int]):
    increments_classes = np.cumsum([0]+increments_classes)
    for old_task, (idx_start, idx_end) in enumerate(zip(increments_classes[:-1], increments_classes[1:])):
        step_idx = np.where((targets>=idx_start)&(targets<idx_end))[0]
        print(f'task{old_task} acc_val : {accuracy_score(np.array(targets)[step_idx], np.array(predictions)[step_idx]):.4f}', end='\t')
    print()

