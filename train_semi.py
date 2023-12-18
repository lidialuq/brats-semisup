# To fix a monai dataloader bug
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import os
import sys
import torch
import wandb

from dataloader import BratsDataset
from torch.utils.data import DataLoader
from monai.losses import DiceLoss

from network import Network
from trainer import Trainer

from monai.networks.nets import UNETR

from transforms_config import transform_train, transform_val

train_samples = int(sys.argv[1])
total_samples = int(sys.argv[2])
print(f'Running with {train_samples} labeled samples and {total_samples} total samples')

config = {
    "name": f"semisup_unet_{train_samples}-{total_samples}",
    "learning_rate": 5e-4,
    "epochs": 1000,
    "train_samples": train_samples,
    "total_samples": total_samples,
    "phi": 0.5,
    "wait_epoch": 15,
    "iterative": True,
    "inputs_pr_iteration": 250,
    "batch_size": 4,
    "scheduler_T_max": 1000,
    "path_data": os.path.join(os.environ['SLURM_TMPDIR'], 'data', 'BraTS_2021'),
    "save_dir": f"./saved_models/semisup_unet_{train_samples}-{total_samples}",
    "save_period": 100,
    "n_gpu": 1,
}
tags = ['semi-supervised', 'size 128', str(train_samples)]

print(config['path_data'])
print(len(os.listdir(config['path_data'])))

    
###############################################################################
# Load Data
###############################################################################
        
# Import dataset and divide 
train_ds = BratsDataset(config['path_data'], split='train', transform=transform_train,
                        samples_label=config['train_samples'])
semi_ds = BratsDataset(config['path_data'], split='semisup', transform=transform_train,
                       samples_label=config['train_samples'], samples_total=config['total_samples'])

val_ds = BratsDataset(config['path_data'], split='val', transform=transform_val)

print(f'Number of labeled samples: {len(train_ds)}, number of unlabeled samples: {len(semi_ds)}, number of validation samples: {len(val_ds)}')

sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=config['inputs_pr_iteration']*config['batch_size'] // 2)
semi_sampler = torch.utils.data.RandomSampler(semi_ds, replacement=True, num_samples=config['inputs_pr_iteration']*config['batch_size'] // 2)

# Make loaders
train_loader = DataLoader(train_ds, batch_size=config['batch_size'] // 2, sampler=sampler,
                        num_workers=8)
semi_loader = DataLoader(semi_ds, batch_size=config['batch_size'] // 2, sampler=semi_sampler,
                        num_workers=8)

val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                        num_workers=8)

loss_function = DiceLoss(include_background=False, softmax=True)

model = Network('unet', 4, 4, (128,128,128))


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=2e-5,
    )

optimizer = torch.optim.Adam(
    model.branch1.parameters(),
    lr=config['learning_rate'],
    weight_decay=2e-5,
    )
optimizer_2 = torch.optim.Adam(
    model.branch2.parameters(),
    lr=config['learning_rate'],
    weight_decay=2e-5,
    )

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, \
    T_max=config['epochs'], eta_min=0, last_epoch=-1, verbose=False)

lr_scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, \
    T_max=config['epochs'], eta_min=0, last_epoch=-1, verbose=False)



trainer = Trainer(
    model=model,
    optimizer=optimizer,
    semi_optimizer=optimizer_2,
    lr_scheduler=lr_scheduler,
    semi_lr_scheduler=lr_scheduler_2,
    loss_function=loss_function,
    config=config,
    project="BratsSemisup",
    data_loader=train_loader,
    semi_supervised_loader=semi_loader,
    valid_data_loader=val_loader,
    seed=None,
    device="cuda:0",
    tags=tags,
    mixed_precision=True,
    phi=config["phi"],
    wait_epoch=config["wait_epoch"]
)

trainer.train()
