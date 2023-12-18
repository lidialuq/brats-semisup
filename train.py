# To fix a monai dataloader bug
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import sys
import os
import torch
import wandb

from dataloader import BratsDataset
from torch.utils.data import DataLoader
from monai.losses import DiceLoss

from network import Network
from trainer import Trainer

from monai.networks.nets import UNETR, BasicUNet

from transforms_config import transform_train, transform_val

train_samples = int(sys.argv[1])
print(f'Running with {train_samples}')


config = {
    "name": f"supervised_unet_{train_samples}",
    "learning_rate": 5e-4,
    "epochs": 1000,
    "train_samples": train_samples,
    "phi": 0.5,
    "wait_epoch": 0,
    "iterative": True,
    "inputs_pr_iteration": 250,
    "batch_size": 4,
    "path_data": os.path.join(os.environ['SLURM_TMPDIR'], 'data', 'BraTS_2021'),
    "save_dir": f"./saved_models/sup_unet_{train_samples}",
    "save_period": 100,
    "n_gpu": 1,
}
tags = ['supervised', 'size 128', str(train_samples)]

    
###############################################################################
# Load Data
###############################################################################
        
# Import dataset and divide 
train_ds = BratsDataset(config['path_data'], split='train', transform=transform_train, 
                        samples_label=config['train_samples'])
val_ds = BratsDataset(config['path_data'], split='val', transform=transform_val)

sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, 
                                         num_samples=config['inputs_pr_iteration']*config['batch_size'])

# Make loaders
train_loader = DataLoader(train_ds, batch_size=config['batch_size'], sampler=sampler,
                        num_workers=8)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                        num_workers=8)

loss_function = DiceLoss(include_background=False, softmax=True)

# model = UNETR(
#     in_channels=4, 
#     out_channels=4, 
#     img_size=(128,128,128), 
#     feature_size=16, 
#     hidden_size=768, 
#     mlp_dim=3072, 
#     num_heads=12, 
#     pos_embed='conv', 
#     norm_name='instance', 
#     conv_block=True, 
#     res_block=True, 
#     dropout_rate=0.0,
#     spatial_dims=3,
#     )

model = BasicUNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    features=(16, 32, 64, 128, 256, 32),
    act='LeakyRELU',
    norm='instance',
    dropout=0.0,
    bias=True,
    upsample='deconv',
)

# print nr of (trainable) parameters
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {pytorch_total_params}")
print(f"Total number of trainable parameters: {pytorch_trainable_params}")

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config['learning_rate'],  
    weight_decay=2e-5,
    )


lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, \
    T_max=config['epochs'], eta_min=0, last_epoch=-1, verbose=False)


trainer = Trainer(
    model=model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    loss_function=loss_function,
    config=config,
    project="BratsSemisup",
    data_loader=train_loader,
    valid_data_loader=val_loader,
    seed=None,
    device="cuda:0",
    tags=tags,
    mixed_precision=True,
    semi_optimizer=None,
    semi_lr_scheduler=None,
    semi_supervised_loader=None,
    phi=None,
)

trainer.train()
