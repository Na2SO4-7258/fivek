import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .config import DATASET,IMAGE_SIZE

@lru_cache(maxsize=None)
def ppsd_users(root):
    users={}
    for path in sorted((Path(root)/"processed").glob("*.json")):
        user=path.stem.rsplit("_",1)[0];rows=users.setdefault(user,{})
        for scene,choice in sorted(json.loads(path.read_bytes()).items()):
            pair=(str(choice["non_prefer"]),str(choice["prefer"]))
            if pair[0]==pair[1]:raise ValueError(f"Identical PPSD preference pair: {path}: {scene}")
            if scene in rows and rows[scene]!=pair:raise ValueError(f"Conflicting PPSD preferences: {user}: {scene}")
            rows[scene]=pair
    if not users:raise FileNotFoundError(f"No PPSD users in {Path(root)/'processed'}")
    return [(user,users[user]) for user in sorted(users)]

def split_ids(root,train_end,end):
    if DATASET=="p":
        count=len(ppsd_users(str(root)))
        if not 0<=train_end<end<count:raise ValueError(f"PPSD user indices require 0 <= train_end < end < {count}; got {train_end}, {end}")
        return range(train_end+1),range(train_end+1,end+1)
    return range(1,train_end+1),range(train_end+1,end+1)

def source_groups(root,cache,ids,experts):
    root=Path(root);cache=Path(cache)
    if DATASET=="f":
        for i in ids:
            d=root/f"{i:04d}";op=d/"O.png"
            if not op.exists():continue
            targets=[(e,d/f"{e}.png") for e in experts if (d/f"{e}.png").exists()]
            if targets:yield cache/f"{i:04d}",op,targets
    else:
        users=ppsd_users(str(root))
        for i in ids:
            if not 0<=i<len(users):raise ValueError(f"PPSD user index {i} outside 0..{len(users)-1}")
            user,rows=users[i]
            for scene,(negative,positive) in sorted(rows.items()):
                d=root/"src"/scene.split("-",1)[0];op=d/f"{scene}_{negative}.jpg";tp=d/f"{scene}_{positive}.jpg"
                for path in (op,tp):
                    if not path.is_file():raise FileNotFoundError(path)
                yield cache/f"{user}_{scene}_{negative}_{positive}",op,[(user,tp)]

def save_npy(p,a):
    tmp=p.with_name(p.stem+".tmp.npy");np.save(tmp,a);tmp.replace(p)

def resize_u8(a):
    x=torch.from_numpy(np.array(a,copy=True)).permute(2,0,1)[None].float()/255.
    x=F.interpolate(x,size=(IMAGE_SIZE,IMAGE_SIZE),mode="area")[0]
    return (x*255+.5).clamp(0,255).byte().permute(1,2,0).numpy()

def build_cache(root,cache,ids,experts,full=False):
    rows=[];groups=list(source_groups(root,cache,ids,experts))
    for c,op,targets in tqdm(groups,desc="cache",unit="image",dynamic_ncols=True):
        c.mkdir(parents=True,exist_ok=True);o=None
        if (full and not (c/"O.npy").exists()) or not (c/"gO.npy").exists():
            with Image.open(op) as im:o=im.convert("RGB");oa=np.asarray(o)
            if full and not (c/"O.npy").exists():save_npy(c/"O.npy",oa)
            if not (c/"gO.npy").exists():save_npy(c/"gO.npy",resize_u8(oa))
        for e,target in targets:
            tp=c/f"{e}.npy";gp=c/f"g{e}.npy"
            if (full and not tp.exists()) or not gp.exists():
                if o is None:
                    with Image.open(op) as im:o=im.convert("RGB")
                with Image.open(target) as im:
                    t=im.convert("RGB")
                    if t.size!=o.size:t=t.resize(o.size,Image.Resampling.LANCZOS)
                    ta=np.asarray(t)
                if full and not tp.exists():save_npy(tp,ta)
                if not gp.exists():save_npy(gp,resize_u8(ta))
            rows.append((c,e))
    return rows
