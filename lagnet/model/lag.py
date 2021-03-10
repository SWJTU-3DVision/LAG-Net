import copy
import os
import time
import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import init
from torchvision.models.resnet import resnet50, Bottleneck, resnet101
#from .resnet import resnet50, Bottleneck
#from .resnet_ibn_a import resnet50_ibn_a, Bottleneck
from .attentions import BasicConv2d, FcWeight, SELayer, LA

EPSILON = 1e-12

def make_model(args):
    return LAG(args)

         
def attention_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal_(m.state_dict()[key], mode='fan_in')
            if 'bn' in key:
                nn.init.constant_(m.state_dict()[key][...], 1.)    
        elif key.split('.')[-1] == 'bias':
            nn.init.constant_(m.state_dict()[key][...], 0.) 
                 

        
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)
   
class LAG(nn.Module):
    def __init__(self, args):
        super(LAG, self).__init__()
        num_classes = args.num_classes
        print(torch.__version__)

        resnet = resnet50(pretrained=True)
        self.test = resnet

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.global_stream = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.part_stream = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.la_stream = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        
        self.maxpool_g_gs = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool_g_ps = nn.AdaptiveMaxPool2d((1, 1))
        self.maxpool_g_la = nn.AdaptiveMaxPool2d(1) 
        self.maxpool_g_d = nn.AdaptiveMaxPool2d(1)       
        self.maxpool_p_ps = nn.AdaptiveMaxPool2d((3, 1))   
        
        self.avgpool_g_gs = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_g_ps = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_g_la = nn.AdaptiveAvgPool2d(1)
        self.avgpool_g_d = nn.AdaptiveMaxPool2d(1) 
        self.avgpool_p_ps = nn.AdaptiveAvgPool2d((3, 1))
        

        reduction = nn.Sequential(nn.Conv2d(2048, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)

        
        
        self.classifier_gs = FcWeight(input_dim=2048, num_classes=num_classes, num_bottleneck=512)        
        self.classifier_ps = FcWeight(input_dim=2048, num_classes=num_classes, num_bottleneck=512)
        self.classifier_la = FcWeight(input_dim=2048, num_classes=num_classes, num_bottleneck=512)
        self.classifier_d = FcWeight(input_dim=2048, num_classes=num_classes, num_bottleneck=512) 
        
        self.classifier_p0_ps = FcWeight(input_dim=2048, num_classes=num_classes, num_bottleneck=512)
        self.classifier_p1_ps = FcWeight(input_dim=2048, num_classes=num_classes, num_bottleneck=512)        
        self.classifier_p2_ps = FcWeight(input_dim=2048, num_classes=num_classes, num_bottleneck=512)   
        
        
        self.M = args.num_attentions
        self.channel_attention = SELayer(2048, 16)
        self.attentions = BasicConv2d(2048, 2048, kernel_size=1)
        self.attentions.apply(attention_init)        
        self.local_attention = LA()
        

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)
   
        
    def getTopK(self, x, K):
        bs, c, h, w = x.data.size() 
        matrix = x.sum(dim=(2,3))
        _, ind = torch.sort(matrix, 1, True)   
        ind = ind.cpu().detach().numpy()  

        attention_map = []
        for i in range(bs):
            attention_map.append(x[i, ind[i][:K], ...])       
        return torch.stack(attention_map)        
            
    def forward(self, x):
    
        x = self.backbone(x)
        batch_size = x.size(0)

        g_s = self.global_stream(x)
        p_s = self.part_stream(x)
        la_s = self.la_stream(x)
        
        # get attention maps
        channel_attention_maps = self.channel_attention(la_s)
        attention_maps = self.attentions(la_s) 
        attention_maps = self.getTopK(attention_maps, self.M) #(b,k,h,w)
        

        feature_vector, feature_matrix = self.local_attention(la_s, attention_maps) #(b, 2048, 6, 1)
        
        # general attention map
        if self.training:
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 1, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)
            d_f = batch_augment(la_s, attention_map, mode='drop', theta=0.5)
        else:
            d_f = la_s            
           
        
        # global feature
        g_gs = self.maxpool_g_gs(g_s) + self.avgpool_g_gs(g_s)
        g_ps = self.maxpool_g_ps(p_s) + self.avgpool_g_ps(p_s)
        g_la = self.maxpool_g_la(feature_vector) + self.avgpool_g_la(feature_vector)
        g_d = self.maxpool_g_d(d_f) + self.avgpool_g_d(d_f)
       
        
        # part feature split
        p_ps = self.maxpool_p_ps(g_ps) + self.avgpool_p_ps(g_ps) #(b, 2048, 3, 1)
        p0_ps = p_ps[:, :, 0:1, :] 
        p1_ps = p_ps[:, :, 1:2, :] 
        p2_ps = p_ps[:, :, 2:3, :] #(b, 2048, 1, 1)
          
        
        # feature for tripletloss
        fg_gs = self.reduction_0(g_gs).squeeze(dim=3).squeeze(dim=2)
        fg_ps = self.reduction_1(g_ps).squeeze(dim=3).squeeze(dim=2)
        fg_la = self.reduction_2(g_la).squeeze(dim=3).squeeze(dim=2)
        fg_d = self.reduction_3(g_d).squeeze(dim=3).squeeze(dim=2)
        
        r0_ps = self.reduction_4(p0_ps)
        r1_ps = self.reduction_5(p1_ps)
        r2_ps = self.reduction_6(p2_ps)
        cat_ps = torch.cat([r0_ps, r1_ps, r2_ps], dim=1)
        f_ps = cat_ps.squeeze(dim=3).squeeze(dim=2) #768
        
        
        #normalize feature matrix
        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)
        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)


        #extract features
        f_gs, l_gs = self.classifier_gs(g_gs.view(batch_size, -1))
        f_ps, l_ps = self.classifier_ps(g_ps.view(batch_size, -1))
        f_la, l_la = self.classifier_la(g_la.view(batch_size, -1))
        f_d, l_d = self.classifier_d(g_d.view(batch_size, -1))
              
        f0_ps, l0_ps = self.classifier_p0_ps(p0_ps.view(batch_size, -1))
        f1_ps, l1_ps = self.classifier_p1_ps(p1_ps.view(batch_size, -1))
        f2_ps, l2_ps = self.classifier_p2_ps(p2_ps.view(batch_size, -1))
        
   
        predict =torch.cat([ f_la, f_d, f_gs, f_ps, f0_ps, f1_ps, f2_ps], dim=1) 
    
        return predict, feature_matrix, fg_gs, fg_ps, fg_la, fg_d, f_ps, l_la, l_d, l_gs, l_ps, l0_ps, l1_ps, l2_ps

        


