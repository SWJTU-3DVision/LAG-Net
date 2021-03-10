import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        #init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
    #init.constant(m.bias.data, 0.0)
            
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True) 
        
class FcWeight(nn.Module):
    def __init__(self, input_dim, num_classes,  relu=True, num_bottleneck=1024):
        super(FcWeight, self).__init__()
        fc_block1 = []
        fc_block2 = []
        fc_block1 += [nn.BatchNorm1d(input_dim)]
        if relu:
            fc_block1 += [nn.LeakyReLU(0.1)]
        fc_block1 += [nn.Linear(input_dim, num_bottleneck, bias = False)] 
        fc_block2 += [nn.BatchNorm1d(num_bottleneck)]
       
        
        fc_block1 = nn.Sequential(*fc_block1)
        fc_block1.apply(weights_init_kaiming)
        fc_block2 = nn.Sequential(*fc_block2)
        fc_block2.apply(weights_init_kaiming)
        
        classifier = []
        classifier += [nn.Linear(num_bottleneck, num_classes, bias = False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.fc_block1 = fc_block1
        self.fc_block2 = fc_block2
        self.classifier = classifier
        
    def forward(self, x):
        x = self.fc_block1(x)
        x1 = self.fc_block2(x)
        x2 = self.classifier(x1)
        return x1, x2        
        
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )   

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)   
        
        
class LA(nn.Module):
    def __init__(self, pool='GAP'):
        super(LA, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)
        #self.pool =
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
    def forward(self, features, attentions):
        B = features.size(0)
        M = attentions.size(1)

        for i in range(M):
            AiF = self.avgpool(features + attentions[:, i:i + 1, ...]) + self.maxpool(features + attentions[:, i:i + 1, ...]) #(b, 2048, 1, 1)
            AiM = AiF.view(B, -1)
            
            if i == 0:
                feature_vector = AiF
                feature_matrix = AiM
            else:
                feature_vector = torch.cat([feature_vector, AiF], dim=2)
                feature_matrix = torch.cat([feature_matrix, AiM], dim=1)

        return feature_vector, feature_matrix             
