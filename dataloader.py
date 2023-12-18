#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:59:53 2022

@author: lidia
"""
from typing import Optional
import os
import nibabel as nib
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from glob import glob

    
class BratsDataset(Dataset):
    """Make dataset for BraTS dataset 2021"""

    def __init__(self, root, split='train', transform=None, 
                 samples_label: Optional[int] = 50, samples_total: Optional[int] = 1050):
        """
        Args:
            root (string): Directory containing all subjects
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Returns: 
            dictionary with keys "path", "image", a (C,H,W,D) tensor and "label", 
            a (H,W,D) tensor with:
                0 = background, 1 = necrosis, 2 = edema, 4 = enhancing
        """
        self.root = root
        self.subjects = [i for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))]
        # sort subjects for reproducibility, then choose split
        self.subjects.sort()
        random.seed(42)
        random.shuffle(self.subjects)
        assert samples_total <= 1050, "samples_total must be less or equal to 1050"
        if split == 'train': 
            self.subjects = self.subjects[:samples_label]
        elif split == 'semisup':
            self.subjects = self.subjects[samples_label:samples_total]
        elif split == 'val':
            self.subjects = self.subjects[1050:1075]
        elif split == 'test':
            self.subjects = self.subjects[1075:]
        
        self.transform = transform
    
    def __len__(self):
        return len(self.subjects)
    
    def __repr__(self):
        return self.subjects

    def __getitem__(self, idx):
        # load sequences into one tensor of shape (C,H,W,D)
        sequences = ['flair', 't1', 't1ce', 't2']
        image = []        
        for seq in sequences: 
            nii = nib.load(os.path.join(self.root, self.subjects[idx], self.subjects[idx]+f'_{seq}.nii.gz'))
            image.append(nii.get_fdata())
        image = np.stack(image, axis=0)
        image = torch.from_numpy(image)
        
        # load label into one tensor of shape (H,W,D)
        nii = nib.load(os.path.join(self.root, self.subjects[idx], self.subjects[idx]+'_seg.nii.gz'))        
        label = torch.from_numpy(nii.get_fdata())
        
        # make dictionary and transform
        sample = {'image': image, 'label': label, 'path': os.path.join(self.root, self.subjects[idx]), 
                'subject': self.subjects[idx]}
        if self.transform:
            sample = self.transform(sample)

        return sample
    

class IvyGap(Dataset):
    """Make dataset for IvyGap(only UPenn annotations, 34 patients"""

    def __init__(self, 
    root='/mnt/CRAI-NAS/all/lidfer/Datasets/Ivy-GAP/Multi-Institutional Paired Expert Segmentations SRI images-atlas-annotations_2/Multi-Institutional Paired Expert Segmentations SRI images-atlas-annotations/1_Images_SRI/CoRegistered_SkullStripped', 
    root_anot='/mnt/CRAI-NAS/all/lidfer/Datasets/Ivy-GAP/Multi-Institutional Paired Expert Segmentations SRI images-atlas-annotations_2/Multi-Institutional Paired Expert Segmentations SRI images-atlas-annotations/3_Annotations_SRI/UPenn',
    root_anot2='/mnt/CRAI-NAS/all/lidfer/Datasets/Ivy-GAP/Multi-Institutional Paired Expert Segmentations SRI images-atlas-annotations_2/Multi-Institutional Paired Expert Segmentations SRI images-atlas-annotations/3_Annotations_SRI/CWRU',
    transform=None, 
    transform2=None):
        """
        Args:
            root (string): Directory containing all subjects with images
            root_anot (string): Directory containing all subjects with annotations
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Returns: 
            dictionary with keys "path", "image", a (C,H,W,D) tensor and "label", 
            a (H,W,D) tensor with (same as Brats):
                0 = background, 1 = necrosis, 2 = edema, 4 = enhancing
        """
        self.root = root
        self.root_anot = root_anot
        self.root_anot2 = root_anot2
        self.subjects = [i for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))]
        self.transform = transform
        self.transform2 = transform2
    
    def __len__(self):
        return len(self.subjects)
    
    def __repr__(self):
        return self.subjects

    def __getitem__(self, idx):
        # load sequences into one tensor of shape (C,H,W,D)
        sequences = ['_flair_', '_t1_', '_t1gd_', '_t2_']
        image = []        
        for seq in sequences: 
            path = glob(os.path.join(self.root, self.subjects[idx], '*', f'*{seq}*.nii.gz'))[0]
            nii = nib.load(path)
            image.append(nii.get_fdata())
        image = np.stack(image, axis=0)
        image = torch.from_numpy(image)
        
        # load label fro uPen into one tensor of shape (H,W,D)
        path = glob(os.path.join(self.root_anot, self.subjects[idx], '*labels.nii.gz'))[0]
        nii = nib.load(path)        
        label = torch.from_numpy(nii.get_fdata())
        # load label from CWRU into one tensor of shape (H,W,D) if it exists, set to none otherwise
        path = glob(os.path.join(self.root_anot2, self.subjects[idx], '*labels.nii.gz'))
        if path:
            nii = nib.load(path[0])        
            label2 = torch.from_numpy(nii.get_fdata())
        else: 
            label2 = 'none'
        
        # make dictionary and transform
        sample = {'image': image, 'label': label, 'label2': label2, 'path': os.path.join(self.root, self.subjects[idx]), 
                'subject': self.subjects[idx], 'path_label': path}
        if self.transform:
            if label2 == 'none':
                sample = self.transform(sample)
            else: 
                sample = self.transform2(sample)

        return sample
