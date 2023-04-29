import argparse
import os
import sys
import yaml
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np

import torch

from utils.metrics import print_num_params

from core.trainer import  train_epoch_adjoint_diff, train_epoch_adjoint_diff_momentum, eval_epoch


from dataset.ImageNet_LT import ImageNetLTDataLoader
from dataset.cifar10 import load_cifar10
from dataset.iNaturalist import INAT
from dataset.cifar100 import load_cifar100
from dataset.cifar_noisy import build_dataloader

import torchvision.models as models
from models.ResNet import ResNet32
from models.VNet import VNet
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config',
                    default="configs/cifar10_noisy/1_ift.yaml", type=str)
args = parser.parse_args()
with open(args.config, mode='r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
device = args["device"]
dataset = args["dataset"]
if dataset == 'Cifar10':
    num_classes = 10
    model = ResNet32(num_classes=num_classes)

    train_dataloader, meta_dataloader, test_dataloader=build_dataloader(dataset="cifar10",num_meta_total=args["num_meta_total"],
                                                                        corruption_type=args["corruption_type"],
                                                                        corruption_ratio=args["corruption_ratio"],
                                                                        batch_size=args["low_batch_size"])
elif dataset == 'Cifar100':
    num_classes = 100
    model = ResNet32(num_classes=num_classes)
    train_dataloader, meta_dataloader, test_dataloader=build_dataloader(dataset="cifar100",num_meta_total=args["num_meta_total"],
                                                                    corruption_type=args["corruption_type"],
                                                                    corruption_ratio=args["corruption_ratio"],
                                                                    batch_size=args["low_batch_size"])

print_num_params(model)

if args["checkpoint"] != 0:
    model = torch.load(f'{args["save_path"]}/epoch_{args.checkpoint}.pth')
    # model.load_state_dict(torch.load(f'{args["save_path"]}/epoch_{args.checkpoint}.pth'))
model = model.to(device)

criterion = nn.CrossEntropyLoss()

vnet=VNet(1,100,1).to(device)


print("train data size",len(train_dataloader.dataset),len(train_dataloader))

if dataset == 'Cifar10' or dataset == 'Cifar100':
    up_start_epoch=args["up_configs"]["start_epoch"]
    if "low_lr_multiplier" in args:
        def warm_up_with_multistep_lr_low(epoch): return (epoch+1) / args["low_lr_warmup"] \
            if epoch < args["low_lr_warmup"] \
            else args["low_lr_multiplier"][len([m for m in args["low_lr_schedule"] if m <= epoch])]
    else:
        def warm_up_with_multistep_lr_low(epoch): return (epoch+1) / args["low_lr_warmup"] \
            if epoch < args["low_lr_warmup"] \
            else 0.1**len([m for m in args["low_lr_schedule"] if m <= epoch])

    def warm_up_with_multistep_lr_up(epoch): return (epoch-up_start_epoch+1) / args["up_lr_warmup"] \
        if epoch-up_start_epoch < args["up_lr_warmup"] \
        else 0.1**len([m for m in args["up_lr_schedule"] if m <= epoch])
        
    train_optimizer = optim.SGD(params=model.parameters(),
                                lr=args["low_lr"], momentum=0.9, weight_decay=1e-4)
    val_optimizer = optim.SGD(params=vnet.parameters(),
                              lr=args["up_lr"], momentum=0.9, weight_decay=1e-4)
    
    train_lr_scheduler = optim.lr_scheduler.LambdaLR(
        train_optimizer, lr_lambda=warm_up_with_multistep_lr_low)
    val_lr_scheduler = optim.lr_scheduler.LambdaLR(
        val_optimizer, lr_lambda=warm_up_with_multistep_lr_up)


if args["save_path"] is None:
    import time
    args["save_path"] = f'./results/{int(time.time())}'
if not os.path.exists(args["save_path"]):
    os.makedirs(args["save_path"])

assert(args["checkpoint"] == 0)

torch.save(model, f'{args["save_path"]}/init_model.pth')
logfile = open(f'{args["save_path"]}/logs.txt', mode='w')
err_log = open(f'{args["save_path"]}/err.txt', mode='w')
with open(f'{args["save_path"]}/config.yaml', mode='w') as config_log:
    yaml.dump(args, config_log)

save_data = {"train_err": [], "val_err": [], "test_err": []}

with open(f'{args["save_path"]}/result.yaml', mode='w') as log:
    yaml.dump(save_data, log)

for i in range(args["checkpoint"], args["epoch"]+1):
    if i % args["checkpoint_interval"] == 0:
        torch.save(model, f'{args["save_path"]}/epoch_{i}.pth')

    if i % args["eval_interval"] == 0:
 
        text, loss, val_err = eval_epoch(meta_dataloader, model, vnet, i, ' val_dataset', args)
        logfile.write(text+'\n')

        text, loss, test_err = eval_epoch(test_dataloader, model, vnet, i, ' test_dataset', args)
        logfile.write(text+'\n')

    save_data["val_err"].append(val_err)
    save_data["test_err"].append(test_err)

    with open(f'{args["save_path"]}/result.yaml', mode='w') as log:
        yaml.dump(save_data, log)

    if args["hg_method"]=='adjoint_diff':
        train_epoch_adjoint_diff(i, model, vnet, args,
                    low_loader=train_dataloader, 
                    low_optimizer=train_optimizer, 
                    up_loader=meta_dataloader, up_optimizer=val_optimizer,

                )
    elif args['hg_method']=='adjoint_diff_momentum':
        train_epoch_adjoint_diff_momentum(i, model, vnet, args,
                low_loader=train_dataloader, 
                low_optimizer=train_optimizer, 
                up_loader=meta_dataloader, up_optimizer=val_optimizer,
                )
    else:
        raise NotImplementedError(args["hg_method"]+"not implemented")
    
  
    err_log.write(f'{val_err} {test_err}\n')
    logfile.flush()
    err_log.flush()
    train_lr_scheduler.step()
    val_lr_scheduler.step()

logfile.close()

err_log.close()
torch.save(model, f'{args["save_path"]}/final_model.pth')
torch.save(vnet, f'{args["save_path"]}/vnet_model.pth')