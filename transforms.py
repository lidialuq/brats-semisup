#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:24:48 2022

@author: lidia
"""

import torch
import numpy as np 
from monai.transforms import MapTransform, Transform

"""
Transforms, who would have guessed. 
"""

class ConvertToMultiChannel(MapTransform):
    """
    Convert label from (H,W,D) to (C,H,W,D) where the channels correspond to 
    (background, edema, enhancing, necrosis). Converts label to float16.
    """
    
    def __call__(self, data):
        d = dict(data)
            
        for key in self.keys:
            result = []
            result.append(d[key]==0) # add background
            result.append(d[key]==2) # add edema
            result.append(d[key]==4) # add enhancing
            result.append(d[key]==1) # add necrosis
            d[key] = torch.stack(result, axis=0).type(torch.HalfTensor)
        return d
