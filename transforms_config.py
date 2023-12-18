import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import monai.transforms as trans
from transforms import ConvertToMultiChannel

size = (128,128,128)
transform_train = trans.Compose([
    ConvertToMultiChannel(keys="label"),
    trans.CropForegroundd(keys=["image", "label"], source_key="image", margin=3, return_coords=False),
    trans.RandZoomd(keys=["image", "label"], prob=0.25, min_zoom=0.9, max_zoom=1.1, mode=('trilinear', 'nearest')),
    trans.RandRotated( keys=['image', 'label'], range_x=0.4, range_y=0.4, range_z=0.4, prob=0.25, padding_mode='zeros', mode=("bilinear", 'nearest')),
    trans.RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=0),
    trans.RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=1),
    trans.RandFlipd(keys=["image", "label"], prob=0.25, spatial_axis=2),
    trans.RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3, spatial_axes=(0, 1)),
    trans.RandAdjustContrastd(keys=["image"], gamma=(0.5, 2), prob=0.1),
    trans.RandHistogramShiftd(keys=["image"], num_control_points=(5, 15), prob=0.1),
    trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    trans.RandStdShiftIntensityd(keys=["image"], factors=0.1, prob=0.1, nonzero=True, channel_wise=True),
    trans.RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.1),
    trans.SpatialPadd(keys=["image", "label"], spatial_size=size, mode='constant'),
    trans.RandSpatialCropd(keys=["image", "label"], roi_size=size, random_center=True, random_size=False),
    trans.EnsureTyped(keys=["image", "label"], data_type='tensor', dtype=torch.float16),
    ])
    
transform_val = trans.Compose([
    ConvertToMultiChannel(keys="label"),
    trans.CropForegroundd(keys=["image", "label"], source_key="image", margin=3, return_coords=False),   
    trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True), 
    trans.SpatialPadd(keys=["image", "label"], spatial_size=size, mode='constant'),
    trans.RandSpatialCropd(keys=["image", "label"], roi_size=size, random_center=True, random_size=False),
    trans.EnsureTyped(keys=["image", "label"], data_type='tensor', dtype=torch.float16),
    ])

transform_test2 = trans.Compose([
    ConvertToMultiChannel(keys=["label", "label2"]),
    trans.CropForegroundd(keys=["image", "label", "label2"], source_key="image", margin=3, return_coords=False),   
    trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True), 
    trans.SpatialPadd(keys=["image", "label", "label2"], spatial_size=size, mode='constant'),
    trans.EnsureTyped(keys=["image", "label", "label2"], data_type='tensor', dtype=torch.float16),
    ])

transform_test = trans.Compose([
    ConvertToMultiChannel(keys="label"),
    trans.CropForegroundd(keys=["image", "label"], source_key="image", margin=3, return_coords=False),   
    trans.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True), 
    trans.SpatialPadd(keys=["image", "label"], spatial_size=size, mode='constant'),
    trans.EnsureTyped(keys=["image", "label"], data_type='tensor', dtype=torch.float16),
    ])
