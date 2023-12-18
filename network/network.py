# encoding: utf-8
from typing import Callable, Dict, Union, Tuple, List, Optional

import torch
import torch.nn as nn
from monai.networks.nets import UNETR, BasicUNet
from torchsummary import summary

class Network(nn.Module):
    def __init__(self, 
                 network_name: str,
                 in_channels: int, 
                 out_channels: int,
                 img_size: Tuple,
                 ):
        super(Network, self).__init__()

        if network_name == 'unetr':
            self.branch1 = UNETR(in_channels=in_channels, 
                                out_channels=out_channels,
                                img_size=img_size, 
                                feature_size=16, 
                                hidden_size=768, 
                                mlp_dim=3072, 
                                num_heads=12,  # Supposed to be 12 
                                pos_embed='conv', 
                                norm_name='instance', 
                                conv_block=True, 
                                res_block=True, 
                                dropout_rate=0.0, 
                                spatial_dims=3) 
            self.branch2 = UNETR(in_channels=in_channels,
                                out_channels=out_channels, 
                                img_size=img_size, 
                                feature_size=16, 
                                hidden_size=768, 
                                mlp_dim=3072, 
                                num_heads=12,  # Supposed to be 12 
                                pos_embed='conv', 
                                norm_name='instance', 
                                conv_block=True, 
                                res_block=True, 
                                dropout_rate=0.0, 
                                spatial_dims=3) 
        elif network_name == 'unet':
            self.branch1 = BasicUNet(spatial_dims=3,
                                    in_channels=4,
                                    out_channels=4,
                                    features=(16, 32, 64, 128, 256, 32),
                                    act='LeakyRELU',
                                    norm='instance',
                                    dropout=0.0,
                                    bias=True,
                                    upsample='deconv')
            self.branch2 = BasicUNet(spatial_dims=3,
                                    in_channels=4,
                                    out_channels=4,
                                    features=(16, 32, 64, 128, 256, 32),
                                    act='LeakyRELU',
                                    norm='instance',
                                    dropout=0.0,
                                    bias=True,
                                    upsample='deconv')

    def forward(self, data, step=1):
        if not self.training:
            pred1 = self.branch1(data)
            pred2 = self.branch2(data)
            return (pred1, pred2)

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)
        


if __name__ == '__main__':
    # test that everything runs
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Network(4, 4, (128,128,128)).to(device)
    left = torch.randn(2, 4, 128, 128, 128).to(device)
    right = torch.randn(2, 4, 128, 128, 128).to(device)
    # summary requires input tensor without batch channel
    summary(model, (4, 128, 128, 128))
    out = model(left)
    print(f'Output shape: {out.shape}')