import MODULES.resnet as resnet
from MODULES.generate_resnet import parse_opts, generate_resnet

import torch
import torch.nn as nn

class FusionNet(nn.Module):
    def __init__(self, img_len=2048, dim_pe=2048, bins=100, 
                 input_D = 100, input_H = 200, input_W = 200):
        super(FusionNet,self).__init__()
        self.input_D = input_D
        self.input_H = input_H
        self.input_W = input_W
        resnet_og, self.parameters = self.__resnet_preparation__()
        lst = list(resnet_og.children())
        self.resnet = nn.Sequential(*lst[0:8])
        self.pool = nn.AdaptiveAvgPool1d(img_len)
        self.fc = nn.Linear(img_len+dim_pe, bins-1)
        

    def forward(self, img, nonimg, out_keys=None):
        out = {}
        x = self.resnet(img)
        x = x.view(x.size(0), -1) # flattern
        x = self.pool (x)
        x = torch.cat([x, nonimg], dim=1)
        out['concatenated'] = x
        out['fc'] = self.fc(x)

        if out_keys:
            return [out[key] for key in out_keys]       
        # return out['fc']
        return out

    def init_weights(self):
        if type(self) == nn.Linear:
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias.data.fill_(0.01)
    
    def __resnet_preparation__(self):
        self.sets = parse_opts()   
        self.sets.pretrain_path = 'pretrain/resnet_10_23dataset.pth'
        self.sets.num_workers = 2
        self.sets.model_depth = 10
        self.sets.input_D = self.input_D
        self.sets.input_H = self.input_H
        self.sets.input_W = self.input_W

        # getting model
        resnet_og, parameters = generate_resnet(self.sets)
        return resnet_og, parameters