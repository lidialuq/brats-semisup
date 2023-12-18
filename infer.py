#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 12:38:56 2022

@author: lidia

Use saved models to predict on BraTS test data.

"""

import copy
import numpy as np
import pickle
from torch.nn.functional import one_hot
from torch.nn import Softmax

import torch
from os.path import join
import os
from tqdm import tqdm
import nibabel as nib
from glob import glob

from dataloader import BratsDataset, IvyGap
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric

from network import Network
from trainer import Trainer

import monai
from monai.networks.nets import UNETR, BasicUNet
from monai.inferers import SlidingWindowInferer

from transforms_config import transform_test

training_data = ['unet_12', 'unet_25', 'unet_50', 'unet_100', 'unet_1050'] #[12, 25, 50, 100, 1050] 
total_data = [1050] #, 600, 150, 'same']

data_folder = "./data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
model_folder =  './saved_models'

predictions_folder = './saved_predictions'
test_results_folder = './saved_test_results'
device = torch.device('cuda:0')


def infer_one(model, data, device, save_folder, semisup, save=False):
    model.eval()
    # Unsqueeze adds batch dimension
    input_volume = data["image"].to(device)
    # Inference and post processing
    with torch.no_grad():
        if semisup:
            output1 = inferer(input_volume.float(), model.branch1)
            output2 = inferer(input_volume.float(), model.branch2)
            output = (softmax(output1) + softmax(output2))/2
        else:
            output = inferer(input_volume.float(), model)
            
    output = torch.argmax(output, dim=1)
    output_onehot = one_hot(output.long(), num_classes=4).permute(0, 4, 1, 2, 3).type(torch.float32).cpu()
    dice = metric(output_onehot, data["label"])
    hausdorff = monai.metrics.compute_hausdorff_distance(output_onehot, data["label"], include_background=False, distance_metric='euclidean', percentile=95)
    
    if save:
        prediction = output.squeeze().detach().cpu().numpy().astype('float32')
        original_nii_path = join(data['path'][0], f"{data['subject'][0]}_seg.nii.gz")
        save_pred = join(save_folder, f"{data['subject'][0]}_prediction.nii.gz")
        # load header and save prediction with old header
        original_nii = nib.load(original_nii_path) #load to get header
        prediction = nib.Nifti1Image(prediction, original_nii.affine)
        nib.save(prediction, save_pred)
        
    return dice[0].numpy(), hausdorff[0].numpy()
        
def save_dic(dice, save_folder):
    dice_dic = {'average': list(),
            'edema': list(),
            'enhancing': list(),
            'necrosis': list(),
            'subjects': list()}
    dice_dic['edema'].append(dice[0])
    dice_dic['enhancing'].append(dice[1])
    dice_dic['necrosis'].append(dice[2])
    dice_dic['average'].append(np.mean(dice))
    dice_dic['subjects'].append(data['subject'][0])
    with open(save_test_results, 'wb') as f:
        pickle.dump(dice_dic, f)

##############################################################################

 
test_ds = BratsDataset(data_folder, split='test', transform=transform_test)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=12)
inferer = SlidingWindowInferer((128,128,128), sw_batch_size=8, overlap=0.25, mode='gaussian')
metric = DiceMetric(include_background=False, reduction='none', ignore_empty='False')
softmax = Softmax(dim=1)

#Infer with semisup models
for tr_data in tqdm(training_data[0:-1]):
    for tot_data in tqdm(total_data):
        if tot_data == 'same':
            tot_data = tr_data

        print(f'Infering on {tr_data}-{tot_data}')
        model = Network('unet', 4, 4, (128,128,128)).to(device)
        semisup_model = join(model_folder, f'semisup_{tr_data}-{tot_data}', 'checkpoint-best.pth')
        model.load_state_dict(torch.load(semisup_model, map_location=device)['state_dict'])
        
        save_folder = join(predictions_folder, f'semisup_{tr_data}-{tot_data}')
        save_test_results = join(test_results_folder, f'semisup_{tr_data}-{tot_data}.pth')
        
        if not os.path.exists(save_folder): os.mkdir(save_folder) 

        dice_dic = {'average_dice': list(),
                    'edema_dice': list(),
                    'enhancing_dice': list(),
                    'necrosis_dice': list(),
                    'average_hausdorff': list(),
                    'edema_hausdorff': list(),
                    'enhancing_hausdorff': list(),
                    'necrosis_hausdorff': list(),
                    'subjects_hausdorff': list(),
                    'subjects': list(),
                    }

        for data in tqdm(test_loader):
            dice, hausdorff = infer_one(model, data, device, save_folder, semisup=True, save=False)
            
            dice_dic['edema_dice'].append(dice[0])
            dice_dic['enhancing_dice'].append(dice[1])
            dice_dic['necrosis_dice'].append(dice[2])
            dice_dic['average_dice'].append(np.mean(dice))
            
            dice_dic['edema_hausdorff'].append(hausdorff[0])
            dice_dic['enhancing_hausdorff'].append(hausdorff[1])
            dice_dic['necrosis_hausdorff'].append(hausdorff[2])
            dice_dic['average_hausdorff'].append(np.mean(hausdorff))
            
            dice_dic['subjects'].append(data['subject'][0])
        
        with open(save_test_results, 'wb') as f:
            pickle.dump(dice_dic, f)

# Infer with supervised models
for tr_data in training_data:
    print(f'Infering on {tr_data}')
    model = BasicUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        features=(16, 32, 64, 128, 256, 32),
        act='LeakyRELU',
        norm='instance',
        dropout=0.0,
        bias=True,
        upsample='deconv').to(device)

    sup_model = join(model_folder, f'sup_{tr_data}', 'checkpoint-best.pth')
    save_folder = join(predictions_folder, f'sup_{tr_data}')
    save_test_results = join(test_results_folder, f'sup_{tr_data}.pth')
    if not os.path.exists(save_folder): os.mkdir(save_folder) 
    model.load_state_dict(torch.load(sup_model, map_location=device)['state_dict'])
    

    dice_dic = {'average_dice': list(),
                'edema_dice': list(),
                'enhancing_dice': list(),
                'necrosis_dice': list(),
                'average_hausdorff': list(),
                'edema_hausdorff': list(),
                'enhancing_hausdorff': list(),
                'necrosis_hausdorff': list(),
                'subjects_hausdorff': list(),
                'subjects': list(),
                }

    for data in tqdm(test_loader):
        dice, hausdorff = infer_one(model, data, device, save_folder, semisup=False, save=False)
        
        dice_dic['edema_dice'].append(dice[0])
        dice_dic['enhancing_dice'].append(dice[1])
        dice_dic['necrosis_dice'].append(dice[2])
        dice_dic['average_dice'].append(np.mean(dice))
        
        dice_dic['edema_hausdorff'].append(hausdorff[0])
        dice_dic['enhancing_hausdorff'].append(hausdorff[1])
        dice_dic['necrosis_hausdorff'].append(hausdorff[2])
        dice_dic['average_hausdorff'].append(np.mean(hausdorff))
        
        dice_dic['subjects'].append(data['subject'][0])

    with open(save_test_results, 'wb') as f:
        pickle.dump(dice_dic, f)


