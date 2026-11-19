# ======================== 本项目数据设置 ========================
RAW_DIR = "data/raw"             # O.png 和 A-E 专家图
USER_ROOT = "data"               # 各分类/user数字/，不读取历史备份
TRAIN_END = 1500                 # 原图数字编号 < 此值用于训练，其余用于验证
VAL_END = None                   # None=全部剩余编号；整数则仅验证 < 此值
# 其余超参数沿用 phase1/config.py；权重保存路径不变。
# ===============================================================
import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from phase1.config import DATASET,ROOT,CACHE,DEVICE,AMP,SEED,ADAM_BETAS,ADAM_EPS,WEIGHT_DECAY,MIN_LR_RATIO,GRAD_CLIP,PSNR_EPS
from phase1.config import Z_DIM,IMAGE_SIZE,P1_TRAIN_END as CONFIG_TRAIN_END,P1_END as CONFIG_END,P1_EPOCHS as EPOCHS,P1_MICRO_BATCH as MICRO_BATCH,P1_ACCUM_STEPS as ACCUM_STEPS,P1_LR as LR,P1_NUM_WORKERS as NUM_WORKERS,P1_PREFETCH as PREFETCH
from phase1.config import P1_WIDTH as WIDTH,P1_LUT_SIZE as LUT_SIZE,P1_EXPERTS as EXPERTS,P1_SMOOTH_WEIGHT as SMOOTH_WEIGHT,P1_IDENTITY_WEIGHT as IDENTITY_WEIGHT,P1_KL_WEIGHT as KL_WEIGHT,P1_FREE_BITS as FREE_BITS,P1_GLOBAL_P as GLOBAL_P,P1_SAVE as SAVE,P1_VAL_BATCH as VAL_BATCH,P1_MSE_WEIGHT as MSE_WEIGHT,P1_FLIP_P as FLIP_P
import math,os,random,json,hashlib,re
from collections import Counter
from PIL import Image,ImageOps
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from phase1.data import source_groups,split_ids,resize_u8
from phase1.model import Model
def resolve(path):
    path=Path(path)
    return (path if path.is_absolute() else PROJECT/path).resolve()
ROOT=resolve(ROOT);CACHE=resolve(CACHE);SAVE=resolve(SAVE)
DEVICE=torch.device(DEVICE if DEVICE!="cuda" or torch.cuda.is_available() else "cpu")
USE_AMP=AMP and DEVICE.type=="cuda"
AMP_DTYPE=torch.bfloat16 if USE_AMP and torch.cuda.is_bf16_supported() else torch.float16
torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
if DEVICE.type=="cuda":torch.backends.cudnn.benchmark=True
def to_t(a):return torch.from_numpy(np.array(a)).permute(2,0,1).float()/255.
def psnr(a,b):
    mse=(a-b).square().flatten(1).mean(1)
    return -10*torch.log10(mse.clamp_min(PSNR_EPS))
def collect_pairs():
    """返回 (原图, 修图, 原图编号, 专家/用户)，仅扫描当前用户目录。"""
    raw=resolve(RAW_DIR);user_root=resolve(USER_ROOT)
    if not raw.is_dir():raise FileNotFoundError(raw)
    rows=[];originals={};seen=set();skipped=0
    for folder in sorted((p for p in raw.iterdir() if p.is_dir() and p.name.isdecimal()),key=lambda p:int(p.name)):
        number=int(folder.name)
        if number in seen:raise ValueError(f"Duplicate original ID: {folder.name}")
        seen.add(number);source=folder/"O.png"
        if not source.is_file():skipped+=1;continue
        originals[number]=source
        rows.extend((source,folder/f"{e}.png",folder.name,e) for e in "ABCDE" if (folder/f"{e}.png").is_file())
    if not user_root.is_dir():raise FileNotFoundError(user_root)
    groups=[user_root]+sorted(p for p in user_root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    for group in groups:
        users=sorted(p for p in group.iterdir() if p.is_dir() and re.fullmatch(r"user[0-9]+",p.name))
        if not users:continue
        sources=dict(originals);manifest=group/"user_parameters.json"
        if manifest.is_file():
            metadata=json.loads(manifest.read_text(encoding="utf-8"));ids=set()
            for item in metadata["images"]:
                number=int(item["id"])
                if number in ids:raise ValueError(f"Duplicate user manifest ID: {manifest}: {number}")
                ids.add(number);sources[number]=group/item["source"]
        for user in users:
            seen=set();style=user.relative_to(user_root).as_posix()
            for target in sorted(user.iterdir()):
                if not target.is_file() or target.suffix.lower() not in (".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp") or not target.stem.isdecimal():continue
                number=int(target.stem)
                if number in seen:raise ValueError(f"Duplicate user image ID: {user}: {number}")
                seen.add(number);source=sources.get(number)
                if source is None or not source.is_file():skipped+=1;continue
                rows.append((source,target,target.stem,style))
    print(f"Available pairs={len(rows)}; missing originals skipped={skipped}",flush=True)
    return rows
def split_rows():
    if DATASET=="f":
        if TRAIN_END<1 or (VAL_END is not None and VAL_END<=TRAIN_END):raise ValueError("Invalid TRAIN_END/VAL_END")
        rows=collect_pairs()
        train=[r for r in rows if int(r[2])<TRAIN_END]
        val=[r for r in rows if int(r[2])>=TRAIN_END and (VAL_END is None or int(r[2])<VAL_END)]
    else:
        train_ids,val_ids=split_ids(ROOT,CONFIG_TRAIN_END,CONFIG_END)
        def pairs(ids):
            return [(source,target,c.name,e) for c,source,targets in source_groups(ROOT,CACHE,ids,EXPERTS) for e,target in targets]
        train,val=pairs(train_ids),pairs(val_ids)
    if not train or not val:raise ValueError("Training and validation sets must both contain image pairs")
    for name,rows in (("train",train),("val",val)):
        print(f"{name}: originals={len(set(r[2] for r in rows))}, pairs={len(rows)}, styles={dict(sorted(Counter(r[3] for r in rows).items()))}",flush=True)
    return train,val
def data_signature(train,val):
    stats={}
    for source,target,_,_ in train+val:
        for path in (source,target):
            if str(path) not in stats:
                st=path.stat();stats[str(path)]=(st.st_size,st.st_mtime_ns)
    payload={"train":[[str(x) for x in r] for r in train],"val":[[str(x) for x in r] for r in val],
             "files":stats,"image_size":IMAGE_SIZE,"alignment":"EXIF/RGB/LANCZOS/area-v1"}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
def read_pair(source,target):
    with Image.open(source) as im:o=ImageOps.exif_transpose(im).convert("RGB")
    with Image.open(target) as im:t=ImageOps.exif_transpose(im).convert("RGB")
    if t.size!=o.size:t=t.resize(o.size,Image.Resampling.LANCZOS)
    return np.asarray(o),np.asarray(t)

class FiveK(Dataset):
    def __init__(self,rows,train):
        self.rows=rows;self.train=train
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        source,target,image_id,e=self.rows[i]
        original,edited=read_pair(source,target)
        go=to_t(resize_u8(original));gt=to_t(resize_u8(edited))
        o,t=go,gt
        if self.train:
            if random.random()>=GLOBAL_P:
                o=original;t=edited
                h,w=o.shape[:2]
                if min(h,w)<IMAGE_SIZE:
                    s=IMAGE_SIZE/min(h,w);nh=max(IMAGE_SIZE,math.ceil(h*s));nw=max(IMAGE_SIZE,math.ceil(w*s))
                    o=F.interpolate(to_t(o)[None],size=(nh,nw),mode="bilinear",align_corners=False)[0]
                    t=F.interpolate(to_t(t)[None],size=(nh,nw),mode="bilinear",align_corners=False)[0]
                    y=random.randint(0,nh-IMAGE_SIZE);x=random.randint(0,nw-IMAGE_SIZE)
                    o=o[:,y:y+IMAGE_SIZE,x:x+IMAGE_SIZE];t=t[:,y:y+IMAGE_SIZE,x:x+IMAGE_SIZE]
                else:
                    y=random.randint(0,h-IMAGE_SIZE);x=random.randint(0,w-IMAGE_SIZE)
                    o=to_t(o[y:y+IMAGE_SIZE,x:x+IMAGE_SIZE]);t=to_t(t[y:y+IMAGE_SIZE,x:x+IMAGE_SIZE])
            if random.random()<FLIP_P:
                go=go.flip(-1);gt=gt.flip(-1);o=o.flip(-1);t=t.flip(-1)
            if random.random()<FLIP_P:
                go=go.flip(-2);gt=gt.flip(-2);o=o.flip(-2);t=t.flip(-2)
        return go.contiguous(),gt.contiguous(),o.contiguous(),t.contiguous(),image_id,e
def lut_smooth(lut):
    a=(lut[:,:,1:,:,:]-lut[:,:,:-1,:,:]).square().mean()
    b=(lut[:,:,:,1:,:]-lut[:,:,:,:-1,:]).square().mean()
    c=(lut[:,:,:,:,1:]-lut[:,:,:,:,:-1]).square().mean()
    return (a+b+c)/3
@torch.inference_mode()
def evaluate(model,loader):
    model.eval()
    losses=[];scores=[];znorms=[];lut_deltas=[];kls=[]
    for go,gt,image,target,_,_ in tqdm(loader,desc="Phase1 validation",unit="batch",dynamic_ncols=True,ascii=True,file=sys.stdout):
        go=go.to(DEVICE,non_blocking=True);gt=gt.to(DEVICE,non_blocking=True)
        image=image.to(DEVICE,non_blocking=True);target=target.to(DEVICE,non_blocking=True)
        with torch.autocast("cuda",dtype=AMP_DTYPE,enabled=USE_AMP):
            out,z,mu,kl,lut,delta=model(go,gt,image)
        out=out.float().clamp(0,1);target=target.float()
        e=out-target
        losses.extend((e.abs()+MSE_WEIGHT*e.square()).flatten(1).mean(1).cpu().tolist())
        scores.extend(psnr(out,target).cpu().tolist())
        znorms.extend(mu.float().norm(dim=1).cpu().tolist())
        lut_deltas.append(delta.float().abs().mean().item())
        kls.append(kl.mean().item())
    return float(np.mean(losses)),float(np.mean(scores)),float(np.mean(znorms)),float(np.mean(lut_deltas)),float(np.mean(kls))
def restore_training(model,opt,scheduler,scaler,signature):
    best=-float("inf");start_epoch=1
    if not SAVE.is_file():return best,start_epoch
    ckpt=torch.load(SAVE,map_location="cpu",weights_only=True)
    model.load_state_dict(ckpt["model"],strict=True)
    if ckpt.get("data_signature")!=signature:
        print(f"Initialize from {SAVE}; dataset changed: reset epoch, optimizer and best score",flush=True)
        return best,start_epoch
    best=float(ckpt.get("val_psnr",best));start_epoch=int(ckpt.get("epoch",0))+1
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
        for state in opt.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):state[k]=v.to(DEVICE)
    if "scheduler" in ckpt:scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("scaler"):scaler.load_state_dict(ckpt["scaler"])
    print(f"resume={SAVE} epoch={start_epoch-1} best={best:.4f}",flush=True)
    return best,start_epoch
def main():
    SAVE.parent.mkdir(parents=True,exist_ok=True)
    train_rows,val_rows=split_rows()
    signature=data_signature(train_rows,val_rows)
    if not train_rows or not val_rows:raise ValueError("Training and validation sets must both contain image pairs")
    train_set=FiveK(train_rows,True);val_set=FiveK(val_rows,False)
    kw=dict(num_workers=NUM_WORKERS,pin_memory=DEVICE.type=="cuda",persistent_workers=NUM_WORKERS>0)
    if NUM_WORKERS>0:kw["prefetch_factor"]=PREFETCH
    train_loader=DataLoader(train_set,batch_size=MICRO_BATCH,shuffle=True,drop_last=False,**kw)
    val_loader=DataLoader(val_set,batch_size=VAL_BATCH,shuffle=False,**kw)
    model=Model(Z_DIM,WIDTH,LUT_SIZE).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=ADAM_BETAS,eps=ADAM_EPS,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=LR*MIN_LR_RATIO)
    scaler=torch.amp.GradScaler("cuda",enabled=USE_AMP and AMP_DTYPE==torch.float16)
    best,start_epoch=restore_training(model,opt,scheduler,scaler,signature)
    print(f"dataset={DATASET} device={DEVICE} amp={USE_AMP}:{AMP_DTYPE} train={len(train_set)} val={len(val_set)} dim={Z_DIM} lut_size={LUT_SIZE} micro={MICRO_BATCH} accum={ACCUM_STEPS} kl_w={KL_WEIGHT} free={FREE_BITS} global_p={GLOBAL_P} params={sum(p.numel() for p in model.parameters()):,}")
    if start_epoch>EPOCHS:
        print(f"checkpoint already reached epoch {start_epoch-1}");return
    for epoch in range(start_epoch,EPOCHS+1):
        model.train()
        losses=[];scores=[];znorms=[];lut_deltas=[];kls=[]
        opt.zero_grad(set_to_none=True)
        bar=tqdm(train_loader,desc=f"Phase1 train {epoch}/{EPOCHS}",unit="batch",leave=True,dynamic_ncols=True,ascii=True,file=sys.stdout)
        for step,(go,gt,image,target,_,_) in enumerate(bar,1):
            go=go.to(DEVICE,non_blocking=True);gt=gt.to(DEVICE,non_blocking=True)
            image=image.to(DEVICE,non_blocking=True);target=target.to(DEVICE,non_blocking=True)
            with torch.autocast("cuda",dtype=AMP_DTYPE,enabled=USE_AMP):
                out,z,mu,kl,lut,delta=model(go,gt,image)
                e=out-target
                image_loss=e.abs().mean()+MSE_WEIGHT*e.square().mean()
                loss=image_loss+SMOOTH_WEIGHT*lut_smooth(lut)+IDENTITY_WEIGHT*delta.square().mean()+KL_WEIGHT*kl.clamp_min(FREE_BITS).mean()
                scaled_loss=loss/ACCUM_STEPS
            scaler.scale(scaled_loss).backward()
            if step%ACCUM_STEPS==0 or step==len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
                scaler.step(opt);scaler.update()
                opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                p=psnr(out.float().clamp(0,1),target.float())
                scores.extend(p.cpu().tolist())
                losses.append(loss.item())
                znorms.extend(mu.float().norm(dim=1).cpu().tolist())
                lut_deltas.append(delta.float().abs().mean().item())
                kls.append(kl.mean().item())
            bar.set_postfix(loss=f"{loss.item():.5f}",psnr=f"{p.mean().item():.3f}")
        val_loss,val_psnr,val_znorm,val_lut_delta,val_kl=evaluate(model,val_loader)
        scheduler.step()
        if val_psnr>best:
            best=val_psnr
            torch.save({"model":model.state_dict(),"encoder":model.encoder.state_dict(),"decoder":model.decoder.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"val_psnr":val_psnr,"dim":Z_DIM,"width":WIDTH,"lut_size":LUT_SIZE,"dataset":DATASET,"experts":tuple(sorted(set(r[3] for r in train_rows+val_rows))),"data_signature":signature,"kl_weight":KL_WEIGHT},SAVE)
        if DEVICE.type=="cuda":torch.cuda.empty_cache()
        print(f"Epoch {epoch:04d}/{EPOCHS} loss={np.mean(losses):.6f} train_PSNR={np.mean(scores):.4f} val_loss={val_loss:.6f} val_PSNR={val_psnr:.4f} z_norm={val_znorm:.4f} lut_delta={val_lut_delta:.5f} kl={np.mean(kls):.4f}/{val_kl:.4f} best={best:.4f}")
if __name__=="__main__":main()