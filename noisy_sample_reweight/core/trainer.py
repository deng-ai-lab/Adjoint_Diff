from pickletools import optimize
import torch.nn.functional as F
from utils.metrics import topk_corrects
import torch
from torch.autograd import grad
import numpy as np
import copy
from core.utils import gather_flat_grad, neumann_hyperstep_preconditioner
from core.utils import get_trainable_hyper_params, assign_hyper_gradient
import torch.optim as optim

    
def train_epoch_adjoint_diff(cur_epoch, model, vnet, args,
                              low_loader,  low_optimizer, 
                              up_loader=None, up_optimizer=None):
    """Performs one epoch of ODE bilevel optimization."""
    num_classes = args["num_classes"]
    ARCH_EPOCH = args["up_configs"]["start_epoch"]
    ARCH_END = args["up_configs"]["end_epoch"]
    ARCH_EPOCH_INTERVAL = args["up_configs"]["epoch_interval"]
    ARCH_INTERVAL = args["up_configs"]["iter_interval"]
    ARCH_TRAIN_SAMPLE = args["up_configs"]["train_batches"]
    ARCH_VAL_SAMPLE = args["up_configs"]["val_batches"]
    ARCH_EULAR_STEP = args['enlar_step_number']
    device = args["device"]
    is_up = (cur_epoch >= ARCH_EPOCH) and (cur_epoch <= ARCH_END) and \
        ((cur_epoch+1) % ARCH_EPOCH_INTERVAL) == 0

    if is_up:
        print('lower lr: ', low_optimizer.param_groups[0]['lr'],
              'upper lr: ', up_optimizer.param_groups[0]['lr'])
        up_iter = iter(up_loader)
        low_iter_alt = iter(low_loader)
    else:
        print('lr: ', low_optimizer.param_groups[0]['lr'])
    model.train()
    total_correct = 0.
    total_sample = 0.
    total_loss = 0.
    num_weights = sum(
        p.numel() for p in model.parameters())

    for cur_iter, (low_data, low_targets) in enumerate(low_loader):
        low_data, low_targets = low_data.to(
            device=device, non_blocking=False), low_targets.to(device=device, non_blocking=False)
        if is_up:
            model.train()
            up_optimizer.zero_grad()
            if cur_iter % ARCH_INTERVAL == 0:

                delta_s = 0.1*ARCH_INTERVAL #low_optimizer.param_groups[0]['lr']*ARCH_INTERVAL
                hyber_grad = 0
                d_val_loss_d_theta=torch.zeros(num_weights,device=device)
                for _ in range(ARCH_VAL_SAMPLE):
                    try:
                        up_data, up_targets = next(up_iter)
                    except StopIteration:
                        up_iter = iter(up_loader)
                        up_data, up_targets = next(up_iter)

                    up_data, up_targets = up_data.to(device=device, non_blocking=False), up_targets.to(
                        device=device, non_blocking=False)
                    model.zero_grad()
                    low_optimizer.zero_grad()
                    up_preds = model(up_data)
                    up_loss = F.cross_entropy(
                        up_preds, up_targets)
                    d_val_loss_d_theta += gather_flat_grad(
                        grad(up_loss, model.parameters(), retain_graph=False))
                d_val_loss_d_theta/=ARCH_VAL_SAMPLE
                model_tilde = copy.deepcopy(model)
                low_optimizer_tilde=optim.SGD(model_tilde.parameters(),lr=1)

                for _ in range(ARCH_EULAR_STEP):
                    d_train_loss_d_theta=torch.zeros(num_weights,device=device)
                    for _ in range(ARCH_TRAIN_SAMPLE):
                        try:
                            low_data_alt, low_targets_alt = next(low_iter_alt)
                        except StopIteration:
                            low_iter_alt = iter(low_loader)
                            low_data_alt, low_targets_alt = next(low_iter_alt)

                        low_data_alt, low_targets_alt = low_data_alt.to(
                            device=device, non_blocking=False), low_targets_alt.to(device=device, non_blocking=False)
                        low_optimizer.zero_grad()
                        low_preds = model_tilde(low_data_alt)
                        
                        low_cost = F.cross_entropy(
                        low_preds, low_targets_alt,reduction='none')
                    
                        low_cost_v = torch.reshape(low_cost, (len(low_cost), 1))
            
                        v_lambda = torch.clamp(vnet(low_cost_v),min=0.0)

                        norm_c = torch.sum(v_lambda)

                        if norm_c != 0:
                            v_lambda_norm = v_lambda / norm_c
                        else:
                            v_lambda_norm = v_lambda
                            
                        low_loss = torch.sum(low_cost_v * v_lambda_norm)
                    
                        d_train_loss_d_theta += gather_flat_grad(
                            grad(low_loss, model_tilde.parameters(), create_graph=True))
                    d_train_loss_d_theta/=ARCH_TRAIN_SAMPLE

                    vector_jacobiao_pro=gather_flat_grad(
                        grad(d_train_loss_d_theta, list(vnet.parameters())+list(model_tilde.parameters()), grad_outputs=d_val_loss_d_theta.view(-1), retain_graph=True, allow_unused=True))
                    
                    r_d_func_d_omega=vector_jacobiao_pro[:-num_weights]
                    hyber_grad -= r_d_func_d_omega*(delta_s/ARCH_EULAR_STEP)
                    r_d_func_d_theta=vector_jacobiao_pro[-num_weights:]
                    d_val_loss_d_theta -= r_d_func_d_theta * \
                        (delta_s/ARCH_EULAR_STEP)

                    low_optimizer_tilde.zero_grad()
                    assign_hyper_gradient(
                        model_tilde.parameters(), d_train_loss_d_theta.detach()*-1*(delta_s/ARCH_EULAR_STEP), num_classes)
                    low_optimizer_tilde.step()

                up_optimizer.zero_grad()
                assign_hyper_gradient(vnet.parameters(), hyber_grad, num_classes)
                up_optimizer.step()

        low_preds = model(low_data)

        # Compute the loss       
        cost = F.cross_entropy(low_preds, low_targets, reduction='none')
        cost_v = torch.reshape(cost, (len(cost), 1))
        
        v_lambda = torch.clamp(vnet(cost_v).detach(),min=0.0)

        norm_c = torch.sum(v_lambda)

        if norm_c != 0:
            v_lambda_norm = v_lambda / norm_c
        else:
            v_lambda_norm = v_lambda
            
        loss = torch.sum(cost_v * v_lambda_norm)
        
        # Perform the backward pass
        low_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        low_optimizer.step()

        # Compute the errors
        mb_size = low_data.size(0)
        ks = [1]
        top1_correct = topk_corrects(low_preds, low_targets, ks)[0]

        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct = top1_correct.item()
        total_correct += top1_correct
        total_sample += mb_size
        total_loss += loss*mb_size
    # Log epoch stats
    print(
        f'Epoch {cur_epoch} :  Loss = {total_loss/total_sample}   ACC = {total_correct/total_sample*100.}')



def train_epoch_adjoint_diff_momentum(cur_epoch, model, vnet, args,
                              low_loader,  low_optimizer, 
                              up_loader=None, up_optimizer=None):
    """Performs one epoch of ODE bilevel optimization."""
    num_classes = args["num_classes"]
    ARCH_EPOCH = args["up_configs"]["start_epoch"]
    ARCH_END = args["up_configs"]["end_epoch"]
    ARCH_EPOCH_INTERVAL = args["up_configs"]["epoch_interval"]
    ARCH_INTERVAL = args["up_configs"]["iter_interval"]
    ARCH_TRAIN_SAMPLE = args["up_configs"]["train_batches"]
    ARCH_VAL_SAMPLE = args["up_configs"]["val_batches"]
    ARCH_EULAR_STEP = args['enlar_step_number']
    device = args["device"]
    is_up = (cur_epoch >= ARCH_EPOCH) and (cur_epoch <= ARCH_END) and \
        ((cur_epoch+1) % ARCH_EPOCH_INTERVAL) == 0

    if is_up:
        print('lower lr: ', low_optimizer.param_groups[0]['lr'],
              'upper lr: ', up_optimizer.param_groups[0]['lr'])
        up_iter = iter(up_loader)
        low_iter_alt = iter(low_loader)
    else:
        print('lr: ', low_optimizer.param_groups[0]['lr'])
    model.train()
    total_correct = 0.
    total_sample = 0.
    total_loss = 0.
    num_weights = sum(
        p.numel() for p in model.parameters())

    for cur_iter, (low_data, low_targets) in enumerate(low_loader):
        low_data, low_targets = low_data.to(
            device=device, non_blocking=False), low_targets.to(device=device, non_blocking=False)
        if is_up:
            model.train()
            up_optimizer.zero_grad()
            if cur_iter % ARCH_INTERVAL == 0:
                momentum_title=torch.cat([i[1]['momentum_buffer'].contiguous().view(-1) for i in low_optimizer.state_dict()['state'].items()])
                M=low_optimizer.param_groups[0]['lr']
                lamda=0.1/low_optimizer.param_groups[0]['lr']
                delta_s = low_optimizer.param_groups[0]['lr']*ARCH_INTERVAL
                #delta_s = 0.1*ARCH_INTERVAL
                hyber_grad = 0
                d_val_loss_d_theta=torch.zeros(num_weights,device=device)
                for _ in range(ARCH_VAL_SAMPLE):
                    try:
                        up_data, up_targets = next(up_iter)
                    except StopIteration:
                        up_iter = iter(up_loader)
                        up_data, up_targets = next(up_iter)

                    up_data, up_targets = up_data.to(device=device, non_blocking=False), up_targets.to(
                        device=device, non_blocking=False)
                    model.zero_grad()
                    low_optimizer.zero_grad()
                    up_preds = model(up_data)
                    up_loss = F.cross_entropy(
                        up_preds, up_targets)
                    d_val_loss_d_theta += gather_flat_grad(
                        grad(up_loss, model.parameters(), retain_graph=False))
                d_val_loss_d_theta/=ARCH_VAL_SAMPLE
                model_tilde = copy.deepcopy(model)
                low_optimizer_tilde=optim.SGD(model_tilde.parameters(),lr=1)

                for _ in range(ARCH_EULAR_STEP):
                    d_train_loss_d_theta=torch.zeros(num_weights,device=device)
                    for _ in range(ARCH_TRAIN_SAMPLE):
                        try:
                            low_data_alt, low_targets_alt = next(low_iter_alt)
                        except StopIteration:
                            low_iter_alt = iter(low_loader)
                            low_data_alt, low_targets_alt = next(low_iter_alt)

                        low_data_alt, low_targets_alt = low_data_alt.to(
                            device=device, non_blocking=False), low_targets_alt.to(device=device, non_blocking=False)
                        low_optimizer.zero_grad()
                        low_preds = model_tilde(low_data_alt)
                        
                        low_cost = F.cross_entropy(
                        low_preds, low_targets_alt,reduction='none')
                    
                        low_cost_v = torch.reshape(low_cost, (len(low_cost), 1))
            
                        v_lambda = torch.clamp(vnet(low_cost_v),min=0.0)

                        norm_c = torch.sum(v_lambda)

                        if norm_c != 0:
                            v_lambda_norm = v_lambda / norm_c
                        else:
                            v_lambda_norm = v_lambda
                            
                        low_loss = torch.sum(low_cost_v * v_lambda_norm)
                    
                        d_train_loss_d_theta += gather_flat_grad(
                            grad(low_loss, model_tilde.parameters(), create_graph=True))
                    d_train_loss_d_theta/=ARCH_TRAIN_SAMPLE

                    vector_jacobiao_pro=gather_flat_grad(
                        grad(d_train_loss_d_theta, list(vnet.parameters())+list(model_tilde.parameters()), grad_outputs=d_val_loss_d_theta.view(-1), retain_graph=True, allow_unused=True))
                    
                    r_d_func_d_omega=vector_jacobiao_pro[:-num_weights]
                    hyber_grad -= r_d_func_d_omega*(delta_s/ARCH_EULAR_STEP)
                    r_d_func_d_theta=vector_jacobiao_pro[-num_weights:]
                    d_val_loss_d_theta -= r_d_func_d_theta * \
                        (delta_s/ARCH_EULAR_STEP)
                    
                    low_optimizer_tilde.zero_grad()
                    assign_hyper_gradient(
                        model_tilde.parameters(), momentum_title*-1*(delta_s/ARCH_EULAR_STEP), num_classes)
                    low_optimizer_tilde.step()

                    momentum_title=(1+lamda*(delta_s/ARCH_EULAR_STEP))*momentum_title-d_train_loss_d_theta.detach()*(delta_s/ARCH_EULAR_STEP)/M
                up_optimizer.zero_grad()
                assign_hyper_gradient(vnet.parameters(), hyber_grad, num_classes)
                torch.nn.utils.clip_grad_norm_(vnet.parameters(),7)
                up_optimizer.step()

        low_preds = model(low_data)

        # Compute the loss       
        cost = F.cross_entropy(low_preds, low_targets, reduction='none')
        cost_v = torch.reshape(cost, (len(cost), 1))
        
        v_lambda = torch.clamp(vnet(cost_v).detach(),min=0.0)

        norm_c = torch.sum(v_lambda)

        if norm_c != 0:
            v_lambda_norm = v_lambda / norm_c
        else:
            v_lambda_norm = v_lambda
            
        loss = torch.sum(cost_v * v_lambda_norm)
        
        # Perform the backward pass
        low_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        low_optimizer.step()

        # Compute the errors
        mb_size = low_data.size(0)
        ks = [1]
        top1_correct = topk_corrects(low_preds, low_targets, ks)[0]

        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct = top1_correct.item()
        total_correct += top1_correct
        total_sample += mb_size
        total_loss += loss*mb_size
    # Log epoch stats
    print(
        f'Epoch {cur_epoch} :  Loss = {total_loss/total_sample}   ACC = {total_correct/total_sample*100.}')


@torch.no_grad()
def eval_epoch(data_loader, model, vnet, cur_epoch, text, args):
    num_classes = args["num_classes"]
    model.eval()
    correct = 0.
    total = 0.
    loss = 0.
    class_correct = np.zeros(num_classes, dtype=float)
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    for cur_iter, (data, targets) in enumerate(data_loader):
        # if cur_iter%5==0:
        #     print(cur_iter,len(data_loader))
        data, targets = data.cuda(), targets.cuda(non_blocking=False)
        logits = model(data)

        preds = logits.data.max(1)[1]
        for t, p in zip(targets.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        mb_size = data.size(0)
        loss += F.cross_entropy(logits, targets).item()*mb_size
    class_correct = confusion_matrix.diag().cpu().numpy()
    total = confusion_matrix.sum().cpu().numpy()
    correct = class_correct.sum()

    text = f'{text}: Epoch {cur_epoch} :  Loss = {loss/total}   ACC = {correct/total*100.}'
    print(text)
    return text, loss/total, 100.-correct/total*100.
