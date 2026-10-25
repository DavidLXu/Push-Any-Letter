"""Independent end-to-end visual encoders for BC, diffusion, flow matching and ACT."""
import torch
from torch import nn
from .models import MODELS


class Residual(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.main=nn.Sequential(nn.Conv2d(cin,cout,3,stride,padding=1,bias=False),
            nn.GroupNorm(8,cout),nn.ReLU(),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout))
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Conv2d(cin,cout,1,stride,bias=False)
    def forward(self,x):return torch.relu(self.main(x)+self.skip(x))


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(3,32,5,stride=2,padding=2,bias=False),nn.GroupNorm(8,32),nn.ReLU(),
            Residual(32,32),Residual(32,32),Residual(32,64,2),Residual(64,64),
            Residual(64,128,2),Residual(128,128),Residual(128,256,2),Residual(256,256),nn.Conv2d(256,32,1))
        yy,xx=torch.meshgrid(torch.linspace(-1,1,6),torch.linspace(-1,1,6),indexing='ij')
        self.register_buffer('locations',torch.stack([xx.flatten(),yy.flatten()],dim=-1))
        self.projection=nn.Sequential(nn.Linear(64,256),nn.ReLU())

    def forward(self, images):
        # Inputs stay uint8 in host memory and are normalized exactly once here.
        x=images.float()/255.
        scores=self.cnn(x).flatten(2)
        keypoints=(scores.softmax(-1)@self.locations).flatten(1)
        return self.projection(keypoints)


class ImagePolicy(nn.Module):
    def __init__(self, algo):
        super().__init__()
        self.encoder=VisualEncoder()
        # 256 visual features + pusher XY + goal XY/sin/cos + 52-way task identity.
        self.policy=MODELS[algo](obs_dim=314)
        self.horizon=self.policy.horizon

    def condition(self, obs):
        images=obs['images']
        b,t,c,h,w=images.shape
        features=self.encoder(images.reshape(b*t,c,h,w)).reshape(b,t,256)
        return torch.cat([features,obs['aux']],dim=-1)

    def loss(self,obs,actions,mask):return self.policy.loss(self.condition(obs),actions,mask)
    def sample(self,obs):return self.policy.sample(self.condition(obs))
