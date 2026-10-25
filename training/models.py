"""Compact state adaptations, not exact reproductions of the published benchmarks."""
import math
import torch
from torch import nn
from .data import OBS_DIM


def masked_mean(error, mask):
    return (error*mask[...,None]).sum() / (mask.sum().clamp_min(1)*error.shape[-1])


class BC(nn.Module):
    horizon = 1
    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(2*obs_dim,256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(), nn.Linear(256,2))

    def sample(self, obs):
        return self.net(obs).unsqueeze(1).clamp(-1,1)

    def loss(self, obs, actions, mask):
        loss = masked_mean((self.net(obs).unsqueeze(1)-actions)**2, mask)
        return loss, {'mse': loss.detach()}


class TimeEmbedding(nn.Module):
    def forward(self, t):
        freqs = torch.exp(torch.arange(32, device=t.device)*(-math.log(10000)/31))
        phase = t[:,None]*1000*freqs
        return torch.cat([phase.sin(),phase.cos()],dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.conv1 = nn.Conv1d(incoming,outgoing,3,padding=1)
        self.norm1 = nn.GroupNorm(8,outgoing)
        self.film = nn.Linear(256,outgoing*2)
        self.conv2 = nn.Conv1d(outgoing,outgoing,3,padding=1)
        self.norm2 = nn.GroupNorm(8,outgoing)
        self.skip = nn.Conv1d(incoming,outgoing,1) if incoming != outgoing else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.norm1(self.conv1(x))
        scale, bias = self.film(cond).chunk(2, dim=-1)
        h = self.act(h*(1+scale[...,None])+bias[...,None])
        h = self.act(self.norm2(self.conv2(h)))
        return h+self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()
        self.time = TimeEmbedding()
        self.condition = nn.Sequential(nn.Linear(64+obs_dim*2,256),nn.SiLU(),nn.Linear(256,256))
        self.enc = FiLMBlock(2,128)
        self.down = nn.Conv1d(128,256,4,stride=2,padding=1)
        self.mid1, self.mid2 = FiLMBlock(256,256), FiLMBlock(256,256)
        self.up = nn.ConvTranspose1d(256,128,4,stride=2,padding=1)
        self.dec = FiLMBlock(256,128)
        self.out = nn.Conv1d(128,2,1)

    def forward(self, x, t, obs):
        cond = self.condition(torch.cat([self.time(t),obs.flatten(1)],dim=-1))
        skip = self.enc(x.transpose(1,2),cond)
        h = self.mid2(self.mid1(self.down(skip),cond),cond)
        h = self.dec(torch.cat([self.up(h),skip],dim=1),cond)
        return self.out(h).transpose(1,2)


class DiffusionPolicy(nn.Module):
    horizon = 16
    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()
        self.net = ConditionalUNet(obs_dim)
        t = torch.linspace(0,1,101,dtype=torch.float64)
        cumulative = torch.cos((t+.008)/1.008*math.pi/2)**2
        cumulative = cumulative/cumulative[0]
        beta = (1-cumulative[1:]/cumulative[:-1]).clamp(max=.999)
        self.register_buffer('abar',torch.cumprod(1-beta,dim=0).float())

    def loss(self, obs, actions, mask):
        t = torch.randint(100,(len(obs),),device=obs.device)
        noise = torch.randn_like(actions)
        a = self.abar[t,None,None]
        noisy = a.sqrt()*actions+(1-a).sqrt()*noise
        error = (self.net(noisy,t.float()/99,obs)-noise)**2
        loss = masked_mean(error,mask)
        return loss, {'epsilon_mse':loss.detach()}

    def sample(self, obs):
        x = torch.randn(len(obs),self.horizon,2,device=obs.device)
        times = torch.linspace(99,0,10,device=obs.device).round().long()
        for j,t in enumerate(times):
            a = self.abar[t]
            prev = self.abar[times[j+1]] if j+1 < len(times) else torch.ones_like(a)
            eps = self.net(x,t.expand(len(obs)).float()/99,obs)
            clean = ((x-(1-a).sqrt()*eps)/a.sqrt()).clamp(-1,1)
            # Recompute epsilon after clipping to keep the DDIM update consistent.
            eps = (x-a.sqrt()*clean)/(1-a).sqrt()
            x = prev.sqrt()*clean+(1-prev).sqrt()*eps
        return x.clamp(-1,1)


class FlowMatching(nn.Module):
    horizon = 16
    def __init__(self, obs_dim=OBS_DIM):
        super().__init__(); self.net = ConditionalUNet(obs_dim)

    def loss(self, obs, actions, mask):
        noise = torch.randn_like(actions)
        t = torch.rand(len(obs),device=obs.device)
        x = (1-t[:,None,None])*noise+t[:,None,None]*actions
        loss = masked_mean((self.net(x,t,obs)-(actions-noise))**2,mask)
        return loss, {'velocity_mse':loss.detach()}

    def sample(self, obs):
        x = torch.randn(len(obs),self.horizon,2,device=obs.device)
        for i in range(10):
            t = torch.full((len(obs),),i/10,device=obs.device)
            x = x+self.net(x,t,obs)/10
        return x.clamp(-1,1)


class ACT(nn.Module):
    horizon = 16
    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()
        self.obs_proj = nn.Linear(obs_dim,128)
        self.act_proj = nn.Linear(2,128)
        self.cls = nn.Parameter(torch.zeros(1,1,128))
        self.posterior_pos = nn.Parameter(torch.randn(1,19,128)*.02)
        layer = lambda: nn.TransformerEncoderLayer(128,4,512,dropout=0.,batch_first=True,activation='gelu')
        self.posterior = nn.TransformerEncoder(layer(),2,enable_nested_tensor=False)
        self.stats = nn.Linear(128,64)
        self.latent_proj = nn.Linear(32,128)
        self.memory_pos = nn.Parameter(torch.randn(1,3,128)*.02)
        self.encoder = nn.TransformerEncoder(layer(),2,enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(1,16,128)*.02)
        dec = nn.TransformerDecoderLayer(128,4,512,dropout=0.,batch_first=True,activation='gelu')
        self.decoder = nn.TransformerDecoder(dec,2)
        self.head = nn.Linear(128,2)

    def predict(self, obs, latent):
        mem = torch.cat([self.latent_proj(latent)[:,None],self.obs_proj(obs)],dim=1)+self.memory_pos
        return self.head(self.decoder(self.queries.expand(len(obs),-1,-1),self.encoder(mem)))

    def loss(self, obs, actions, mask):
        tokens = torch.cat([self.cls.expand(len(obs),-1,-1),self.obs_proj(obs),self.act_proj(actions)],dim=1)
        padding = torch.cat([torch.zeros(len(obs),3,device=obs.device,dtype=torch.bool),~mask],dim=1)
        cls = self.posterior(tokens+self.posterior_pos,src_key_padding_mask=padding)[:,0]
        mu, logvar = self.stats(cls).chunk(2,dim=-1)
        logvar = logvar.clamp(-10,10)
        latent = mu+torch.exp(.5*logvar)*torch.randn_like(mu)
        reconstruction = masked_mean((self.predict(obs,latent)-actions).abs(),mask)
        kl = (-.5*(1+logvar-mu.square()-logvar.exp()).sum(-1)).mean()
        return reconstruction+10*kl, {'l1':reconstruction.detach(),'kl':kl.detach()}

    def sample(self, obs):
        return self.predict(obs,torch.zeros(len(obs),32,device=obs.device)).clamp(-1,1)


MODELS = {'bc':BC,'diffusion':DiffusionPolicy,'flow':FlowMatching,'act':ACT}
