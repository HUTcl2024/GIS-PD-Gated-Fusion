"""Single-sensor Swin-T; original parameter names preserved for saved weights."""
from torch import nn
from torchvision import models

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
    if method in ['swin_uhf','swin_hfct']:
        return RecoveredSwin(method.split('_')[1],classes,pretrained)
    raise ValueError('Only single-sensor Swin-T is retained')
