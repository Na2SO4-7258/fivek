import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from .config import Z_DIM as DIM,NORM_EPS,P1_WIDTH as WIDTH,P1_LUT_SIZE as LUT_SIZE,P1_LUT_BASES as NB,P1_KNOTS as KNOTS,P1_ENC_RES as ENC_RES,P1_CM as CM
from .config import P1_DEPTHS as DEPTHS,P1_CM_HIDDEN as CM_HIDDEN,P1_CM_DIM as CM_DIM,P1_HEAD_HIDDEN as HEAD_HIDDEN,P1_LOGVAR_INIT as LOGVAR_INIT,P1_LOGVAR_RANGE as LOGVAR_RANGE
from .config import P1_LUT_HIDDEN as LUT_HIDDEN,P1_LUT_INIT_STD as LUT_INIT_STD,P1_LUT_SCALE as LUT_SCALE,P1_ADJUST_HIDDEN as ADJUST_HIDDEN,P1_CURVE_SCALE as CURVE_SCALE,P1_MATRIX_SCALE as MATRIX_SCALE,P1_BIAS_SCALE as BIAS_SCALE
import torch
from torch import nn
import torch.nn.functional as F

class LN(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.w=nn.Parameter(torch.ones(1,c,1,1))
        self.b=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        m=x.mean(1,keepdim=True);v=(x-m).square().mean(1,keepdim=True)
        return (x-m)/torch.sqrt(v+NORM_EPS)*self.w+self.b

class Block(nn.Module):
    def __init__(self,c,cd=0):
        super().__init__()
        self.n1=LN(c);self.n2=LN(c)
        self.a=nn.Conv2d(c,c*2,1)
        self.dw=nn.Conv2d(c*2,c*2,3,1,1,groups=c*2)
        self.ca=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,c,1))
        self.b=nn.Conv2d(c,c,1)
        self.c=nn.Conv2d(c,c*2,1)
        self.d=nn.Conv2d(c,c,1)
        self.beta=nn.Parameter(torch.zeros(1,c,1,1))
        self.gamma=nn.Parameter(torch.zeros(1,c,1,1))
        self.film=None
        if cd:
            self.film=nn.Linear(cd,c*4)
            nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
    def forward(self,x,z=None):
        if self.film is not None:s1,h1,s2,h2=self.film(z)[:,:,None,None].chunk(4,1)
        n=self.n1(x)
        if self.film is not None:n=n*(1+s1)+h1
        y=self.a(n);p,q=self.dw(y).chunk(2,1);y=p*q;y=y*self.ca(y);y=self.b(y);x=x+self.beta*y
        n=self.n2(x)
        if self.film is not None:n=n*(1+s2)+h2
        p,q=self.c(n).chunk(2,1)
        return x+self.gamma*self.d(p*q)

class Stage(nn.Module):
    def __init__(self,c,n,cd=0):
        super().__init__()
        self.blocks=nn.ModuleList([Block(c,cd) for _ in range(n)])
    def forward(self,x,z=None):
        for b in self.blocks:x=b(x,z)
        return x

def cmap_stat(o,t,n=CM):
    o=o.float();t=t.float();B=o.shape[0]
    i=(o.clamp(0,.9999)*n).long()
    k=(i[:,0]*n*n+i[:,1]*n+i[:,2]).flatten(1)
    s=torch.zeros(B,3,n**3,device=o.device).scatter_add_(2,k[:,None].expand(-1,3,-1),(t-o).flatten(2))
    c=torch.zeros(B,n**3,device=o.device).scatter_add_(1,k,torch.ones_like(k,dtype=torch.float))
    return torch.cat((s/c.clamp_min(1)[:,None],(c/k.shape[1]).sqrt()[:,None]),1).flatten(1)

class EditEncoder(nn.Module):
    def __init__(self,dim=DIM,w=WIDTH):
        super().__init__()
        self.dim=dim
        self.stem=nn.Sequential(nn.Conv2d(9,w,5,2,2),nn.GELU())
        self.s1=Stage(w,DEPTHS[0])
        self.d1=nn.Conv2d(w,w*2,3,2,1)
        self.s2=Stage(w*2,DEPTHS[1])
        self.d2=nn.Conv2d(w*2,w*4,3,2,1)
        self.s3=Stage(w*4,DEPTHS[2])
        self.d3=nn.Conv2d(w*4,w*8,3,2,1)
        self.s4=Stage(w*8,DEPTHS[3])
        self.pool=nn.Sequential(LN(w*8),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.cm=nn.Sequential(nn.LayerNorm(4*CM**3),nn.Linear(4*CM**3,CM_HIDDEN),nn.GELU(),nn.Linear(CM_HIDDEN,CM_DIM))
        self.head=nn.Sequential(nn.Linear(w*8+CM_DIM,HEAD_HIDDEN),nn.GELU(),nn.Linear(HEAD_HIDDEN,dim*2))
        with torch.no_grad():self.head[-1].bias[dim:]=LOGVAR_INIT
    def forward(self,o,t):
        s=cmap_stat(o,t)
        if tuple(o.shape[-2:])!=(ENC_RES,ENC_RES):
            o=F.interpolate(o,size=(ENC_RES,ENC_RES),mode="area");t=F.interpolate(t,size=(ENC_RES,ENC_RES),mode="area")
        x=torch.cat((o,t,t-o),1)
        x=self.s1(self.stem(x));x=self.s2(self.d1(x));x=self.s3(self.d2(x));x=self.s4(self.d3(x))
        f=torch.cat((self.pool(x),self.cm(s)),1)
        mu,lv=self.head(f).float().chunk(2,1);lv=lv.clamp(*LOGVAR_RANGE)
        z=mu+torch.randn_like(mu)*(.5*lv).exp() if self.training else mu
        kl=(.5*(mu.square()+lv.exp()-1-lv)).mean(0)
        return z,mu,kl

def make_identity_lut(size):
    r=torch.linspace(0,1,size)
    rr,gg,bb=torch.meshgrid(r,r,r,indexing="ij")
    return torch.stack((rr,gg,bb),0)

class LUTGenerator(nn.Module):
    def __init__(self,dim=DIM,size=LUT_SIZE,nb=NB):
        super().__init__()
        self.size=size
        self.coef=nn.Sequential(nn.Linear(dim,LUT_HIDDEN),nn.GELU(),nn.Linear(LUT_HIDDEN,nb))
        nn.init.zeros_(self.coef[-1].weight);nn.init.zeros_(self.coef[-1].bias)
        self.basis=nn.Parameter(torch.randn(nb,3*size**3)*LUT_INIT_STD)
        self.register_buffer("identity",make_identity_lut(size))
    def forward(self,z):
        B=z.shape[0]
        d=(self.coef(z)@self.basis).view(B,3,self.size,self.size,self.size)
        d=torch.tanh(d)*LUT_SCALE
        lut=self.identity[None]+d
        return lut.clamp(0,1),d

class GlobalAdjust(nn.Module):
    def __init__(self,dim=DIM,k=KNOTS):
        super().__init__()
        self.k=k
        self.net=nn.Sequential(nn.Linear(dim,ADJUST_HIDDEN),nn.GELU(),nn.Linear(ADJUST_HIDDEN,3*k+12))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("knots",torch.linspace(0,1,k))
        self.register_buffer("eye",torch.eye(3))
    def forward(self,image,z):
        with torch.autocast("cuda",enabled=False):
            B,_,H,W=image.shape;k=self.k
            p=self.net(z.float())
            cv=self.knots+CURVE_SCALE*torch.tanh(p[:,:3*k].view(B,3,k))
            m=self.eye+MATRIX_SCALE*torch.tanh(p[:,3*k:3*k+9].view(B,3,3))
            b=BIAS_SCALE*torch.tanh(p[:,3*k+9:])
            pos=image.float().clamp(0,1)*(k-1)
            i0=pos.floor().clamp(0,k-2)
            f=(pos-i0).flatten(2)
            i0=i0.long().flatten(2)
            y=(cv.gather(2,i0)*(1-f)+cv.gather(2,i0+1)*f).view(B,3,H,W)
            y=torch.einsum("bij,bjhw->bihw",m,y)+b[:,:,None,None]
            return y.clamp(0,1)

def apply_lut(image,lut):
    H,W=image.shape[-2:]
    grid=torch.stack((image[:,2]*2-1,image[:,1]*2-1,image[:,0]*2-1),-1)[:,None]
    out=F.grid_sample(lut,grid,mode="bilinear",padding_mode="border",align_corners=True)
    return out[:, :, 0]

class LUTDecoder(nn.Module):
    def __init__(self,dim=DIM,size=LUT_SIZE):
        super().__init__()
        self.generator=LUTGenerator(dim,size)
        self.adjust=GlobalAdjust(dim)
    def forward(self,image,z):
        lut,delta=self.generator(z)
        out=apply_lut(self.adjust(image,z),lut)
        return out,lut,delta

class Model(nn.Module):
    def __init__(self,dim=DIM,width=WIDTH,lut_size=LUT_SIZE):
        super().__init__()
        self.encoder=EditEncoder(dim,width)
        self.decoder=LUTDecoder(dim,lut_size)
    def encode(self,o,t):return self.encoder(o,t)[0]
    def decode(self,image,z):return self.decoder(image,z)
    def forward(self,go,gt,image):
        z,mu,kl=self.encoder(go,gt)
        out,lut,delta=self.decoder(image,z)
        return out,z,mu,kl,lut,delta