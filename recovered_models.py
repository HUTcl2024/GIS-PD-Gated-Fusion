"""Executable adaptations of architectures recovered from 数据处理.

Only model definitions are reused. Splits, preprocessing, selection and metrics
come from revision_train.py. Pretraining is explicit; download failure raises
instead of silently switching Swin-T to random initialization.
Original source identities and adaptation scope: docs/BASELINES.md.
"""
import torch
from torch import nn
from torchvision import models


def resnet(backbone,pretrained):
    weights=(models.ResNet50_Weights.IMAGENET1K_V1 if backbone=='resnet50'
             else models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else None
    base=getattr(models,backbone)(weights=weights)
    n=base.fc.in_features; base.fc=nn.Identity()
    return base,n


class RecoveredConcat(nn.Module):
    """fusion_concat.py:282-327: two projections + original dropout/MLP head."""
    def __init__(self,backbone,dim,classes,pretrained):
        super().__init__()
        self.uhf_backbone,n=resnet(backbone,pretrained)
        self.hfct_backbone,_=resnet(backbone,pretrained)
        self.uhf_proj=nn.Sequential(nn.Linear(n,dim),nn.ReLU())
        self.hfct_proj=nn.Sequential(nn.Linear(n,dim),nn.ReLU())
        self.classifier=nn.Sequential(nn.Dropout(.5),nn.Linear(dim*2,512),nn.ReLU(),
                                      nn.Dropout(.5),nn.Linear(512,classes))

    def forward(self,u,h):
        features=torch.cat([self.uhf_proj(self.uhf_backbone(u)),
                            self.hfct_proj(self.hfct_backbone(h))],1)
        return self.classifier(features),None


class RecoveredEqual(nn.Module):
    """dqrh.py:282-323: dropout + linear per channel, jointly averaged logits."""
    def __init__(self,backbone,classes,pretrained):
        super().__init__()
        self.uhf_backbone,n=resnet(backbone,pretrained)
        self.hfct_backbone,_=resnet(backbone,pretrained)
        self.uhf_head=nn.Sequential(nn.Dropout(.5),nn.Linear(n,classes))
        self.hfct_head=nn.Sequential(nn.Dropout(.5),nn.Linear(n,classes))

    def forward(self,u,h):
        return (self.uhf_head(self.uhf_backbone(u))+self.hfct_head(self.hfct_backbone(h)))/2,None


class RecoveredSwin(nn.Module):
    """swin_tyzs.py:74-92: Swin-T + 768->512->ReLU->Dropout->C."""
    def __init__(self,channel,classes,pretrained):
        super().__init__()
        self.channel=channel
        self.model=models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        n=self.model.head.in_features
        self.model.head=nn.Sequential(nn.Linear(n,512),nn.ReLU(),nn.Dropout(.5),nn.Linear(512,classes))

    def forward(self,u,h):
        return self.model(u if self.channel=='uhf' else h),None


def build_recovered(method,backbone,dim,classes,pretrained):
    if method=='recovered_concat': return RecoveredConcat(backbone,dim,classes,pretrained)
    if method=='recovered_equal': return RecoveredEqual(backbone,classes,pretrained)
    if method in ['swin_uhf','swin_hfct']: return RecoveredSwin(method.split('_')[1],classes,pretrained)
    raise ValueError(method)
