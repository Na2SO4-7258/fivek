# ========================= 超参数 =========================
USER_DIR = r"data\random1\user1"       # 或 "A"/"B"/"C"/"D"/"E"
RAW_DIR = r"data\raw"
EXPERT_GROUP_DIR = "data/expert_groups"
PHASE1_CHECKPOINT = r"phase1\out\model\best.pt"

# 首次加载 DINOv2 需联网下载 GitHub Hub 代码及官方预训练权重；以后使用 torch hub 缓存。
DINO_REPO = "facebookresearch/dinov2"
DINO_MODEL = "dinov2_vitb14"        # ViT-B/14，无 registers
DINO_HUB_SOURCE = "github"         # 离线可改成 "local" 并把 DINO_REPO 改成本地 dinov2 仓库路径
# DINO backbone 全量微调：patch_embed、位置编码、tokens、12 个 blocks、norm 全部解冻。
IMAGE_SIZE = 224                   # ViT/14 要求 14 的整数倍；可尝试 280/336（显存增加）
EMBEDDING_DIM = 128
PROJECTION_HIDDEN_DIM = 256       # 非线性投影：768 -> 256 -> GELU -> 128，不做 L2 归一化

TRAIN_END = 1500
VAL_END = 2000
EPOCHS = 100
BATCH_SIZE = 16                   # 显存不足可调至 8（同时适当提高梯度累积）
ACCUMULATION_STEPS = 2            # micro-batch 梯度累积；不会产生跨 micro-batch 的图片对
EVAL_BATCH_SIZE = 16
NUM_WORKERS = 4                   # Windows 异常时设 0
SEED = 42
DEVICE = "cuda"
USE_AMP = True
AMP_DTYPE = "float16"             # 也可选 bfloat16（需 GPU 支持）
BACKBONE_LR = 1e-5
PROJECTION_LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# 采样：每批选择若干随机 anchor，并优先加入其 z 邻居，剩余位置随机补齐
Z_NEIGHBOR_TOP_N = 32
Z_NEIGHBOR_FRACTION = 0.5         # 每批约一半图来自 anchor 邻域
# 距离损失：所有非自身样本对均参与，仅近距离的对有较高权重
LOSS_TAU = 0.5                    # z 距离尺度相关；可根据输出的 z 距离统计微调
LOSS_WEIGHT_FLOOR = 0.1          # 避免远距离监督权重趋近于零
# KNN：Top1 与 Top5 都评估，以 Top5 PSNR 选 best
TOPK = 5
TOPK_TEMPERATURE = 0.25          # 编码距离 softmax 权重温度
BEST_METRIC = "top5_psnr"        # "top1_psnr" 亦可

SAVE_EVERY_EPOCH = True
SAVE_VAL_VISUAL = True
VISUAL_DIR = "out/visual"
OUTPUT_FOLDER = "dinov2_full_mlp_z"  # 独立保存全解冻+MLP 实验，保留之前 dinov2_z 的结果
# ==========================================================
"""
运行：python train.py
依赖：torch、torchvision、numpy、Pillow，及项目原有 phase1、analyze_z.py、user.py。
训练：全量解冻 DINOv2 ViT-B/14 -> Linear(768,256) -> GELU -> Linear(256,128) -> 全对 z 距离加权 MSE。
新投影头与旧线性投影 checkpoint 不兼容；从官方预训练 DINOv2 权重开始，新结果写入 dinov2_full_mlp_z。
样本：z-aware batch sampling；无唯一正样本、无负样本标签、无 InfoNCE。
加权目标：mean(weight_ij * (||e_i-e_j|| - ||z_i-z_j||)^2) / mean(weight_ij)
weight_ij = LOSS_WEIGHT_FLOOR + (1 - LOSS_WEIGHT_FLOOR) * exp(-||z_i-z_j|| / LOSS_TAU)
所有对参与监督；输出不 L2 normalize，保持欧氏距离尺度。
评估：验证原图只用于查询，训练集构建库。Top1 最近训练 z、Top5 距离 softmax 融合 z，分别计算真实 z MSE 和 Phase1 PSNR。
按 Top5 PSNR 选 best.pt；保存所有 val 的 Top5 两行修图图、CSV 与嵌入检索库。
原有专家/模拟用户 z 缓存、四项基线、训练/验证划分不变。
首次 torch.hub 需要联网；网络/环境不允许下载时须事先缓存官方模型和代码。
"""



import csv

import hashlib

import os

import json

import math

import random

from pathlib import Path

from datetime import datetime, timezone
from contextlib import nullcontext



import numpy as np

from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch

from torch import nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Sampler

from torchvision.transforms import functional as TF





ROOT = Path(__file__).resolve().parent

IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)





def resolve(path):

    p = Path(path)

    return (p if p.is_absolute() else ROOT / p).resolve()





def rgb_image(path):

    with Image.open(path) as im:

        return ImageOps.exif_transpose(im).convert("RGB")





def image_input(img):

    x = TF.to_tensor(TF.resize(img, [IMAGE_SIZE, IMAGE_SIZE]))

    return TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)





def expert_code():

    value = str(USER_DIR).strip().upper()

    return value if value in ("A", "B", "C", "D", "E") else None





def load_records(user_dir):

    root = user_dir.parent

    npz = root / "user_latents.npz"

    manifest = root / "user_parameters.json"

    if not npz.is_file() or not manifest.is_file():

        raise FileNotFoundError(f"找不到 {npz} 或 {manifest}，请先运行 user.py")



    metadata = json.loads(manifest.read_text(encoding="utf-8"))

    with np.load(npz, allow_pickle=False) as data:

        users = data["users"].astype(str).tolist()

        ids = data["ids"].astype(str).tolist()

        zs = data["z"].astype(np.float32)



    if user_dir.name not in users:

        raise ValueError(f"{user_dir.name} 不在缓存用户列表中：{users}")

    user_idx = users.index(user_dir.name)

    if zs.shape != (len(users), len(ids), 3):

        raise ValueError(f"z 形状不正确：{zs.shape}")

    if len(metadata["images"]) != len(ids):

        raise ValueError("缓存图片数量与元数据数量不一致")

    if not (0 < TRAIN_END < VAL_END):

        raise ValueError("训练/验证编号边界不合法")



    train_records, val_records = [], []

    seen = set()

    for i, (image_id, item) in enumerate(zip(ids, metadata["images"])):

        if str(item["id"]) != image_id:

            raise ValueError(f"第 {i} 张图片的缓存编号与元数据不一致")

        if not image_id.isdecimal():

            raise ValueError(f"图片编号必须是数字：{image_id}")

        image_num = int(image_id)

        if image_num in seen:

            raise ValueError(f"重复的图片数字编号：{image_id}")

        seen.add(image_num)

        if not (0 <= image_num < VAL_END):

            continue

        source = root / item["source"]

        target = user_dir / f"{image_id}.png"

        if not source.is_file() or not target.is_file():

            raise FileNotFoundError(f"原图或用户修图缺失：{source} / {target}")

        record = (source, target, zs[user_idx, i].copy(), image_id)

        if image_num < TRAIN_END:

            train_records.append(record)

        else:

            val_records.append(record)



    train_records.sort(key=lambda r: int(r[3]))

    val_records.sort(key=lambda r: int(r[3]))

    if not train_records or not val_records:

        raise ValueError(f"按编号划分后训练/验证集为空：训练 {len(train_records)}、验证 {len(val_records)}")

    return train_records, val_records





def read_pair(source, target):

    original, edited = rgb_image(source), rgb_image(target)

    if edited.size != original.size:

        edited = edited.resize(original.size, Image.Resampling.LANCZOS)

    return np.asarray(original, dtype=np.uint8), np.asarray(edited, dtype=np.uint8)





def file_sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        for chunk in iter(lambda: f.read(1024 * 1024), b""):

            h.update(chunk)

    return h.hexdigest()





def expert_inputs(root):

    folders = sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdecimal()),

                     key=lambda p: int(p.name))

    rows, signature, seen = [], [], set()

    for folder in folders:

        if int(folder.name) in seen:

            raise ValueError(f"重复的图片数字编号：{folder.name}")

        seen.add(int(folder.name))

        source = folder / "O.png"

        targets = {e: folder / f"{e}.png" for e in "ABCDE" if (folder / f"{e}.png").is_file()}

        if not source.is_file() or not targets:

            continue

        rows.append((folder.name, source, targets))

        for path in [source, *targets.values()]:

            st = path.stat()

            signature.append((path.relative_to(root).as_posix(), st.st_size, st.st_mtime_ns))

    fingerprint = hashlib.sha256(json.dumps(signature).encode("utf-8")).hexdigest()

    return rows, fingerprint





def expert_cache_context(root):

    rows, signature = expert_inputs(root)

    checkpoint = resolve(PHASE1_CHECKPOINT)

    code_files = (Path(__file__).parent / "phase1/model.py", Path(__file__).parent / "phase1/config.py")

    encoder_signature = hashlib.sha256("".join(file_sha256(p) for p in code_files).encode()).hexdigest()

    context = {"input_signature": signature, "checkpoint_sha256": file_sha256(checkpoint),

               "encoder_signature": encoder_signature, "cache_version": 1,

               "alignment": "EXIF transpose, RGB, expert resized to original with PIL LANCZOS"}

    return rows, context





@torch.inference_mode()

def ensure_expert_cache(phase1, device, force=False):

    root = resolve(RAW_DIR)

    if not root.is_dir():

        raise FileNotFoundError(f"专家数据目录不存在：{root}")

    rows, context = expert_cache_context(root)

    if not rows:

        raise ValueError(f"没有完整的 O/专家图片对：{root}")

    npz_path, metadata_path = root / "user_latents.npz", root / "user_parameters.json"

    if not force and npz_path.is_file() and metadata_path.is_file():

        try:

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            with np.load(npz_path, allow_pickle=False) as data:

                valid = data["valid"]

                z = data["z"]

                if (all(metadata.get(k) == v for k, v in context.items())

                        and metadata.get("latent_sha256") == file_sha256(npz_path)

                        and data["users"].tolist() == list("ABCDE")

                        and data["ids"].tolist() == [r[0] for r in rows]

                        and z.shape == (5, len(rows), 3) and valid.shape == (5, len(rows))

                        and valid.dtype == np.bool_

                        and np.array_equal(valid, np.asarray([[e in r[2] for r in rows] for e in "ABCDE"]))

                        and np.isfinite(z[valid]).all()):

                    print(f"复用真实专家三维 z 缓存：{npz_path}", flush=True)

                    return npz_path, metadata_path

        except (OSError, ValueError, KeyError, TypeError):

            pass



    phase1.eval()

    z = np.full((5, len(rows), 3), np.nan, dtype=np.float32)

    fit_psnr = np.full((5, len(rows)), np.nan, dtype=np.float32)

    valid = np.zeros((5, len(rows)), dtype=bool)

    image_records = []

    for i, (image_id, source, targets) in enumerate(rows):

        original = rgb_image(source)

        names, images = list(targets), []

        for name in names:

            edited = rgb_image(targets[name])

            if edited.size != original.size:

                edited = edited.resize(original.size, Image.Resampling.LANCZOS)

            images.append(np.asarray(edited, dtype=np.uint8))

        raw = np.asarray(original, dtype=np.uint8)

        source_tensor = torch.from_numpy(raw.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.

        source_batch = source_tensor.expand(len(names), -1, -1, -1).contiguous()

        target_batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device).float() / 255.

        latent = phase1.encode(source_batch, target_batch)

        if tuple(latent.shape) != (len(names), 3) or not torch.isfinite(latent).all():

            raise ValueError(f"图片 {image_id} 的 phase1 三维编码无效")

        rendered = phase1.decode(source_batch, latent)[0]

        if not torch.isfinite(rendered).all():

            raise ValueError(f"图片 {image_id} 的 phase1 重建无效")

        pixels = (rendered.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()

        for j, name in enumerate(names):

            u = "ABCDE".index(name)

            z[u, i] = latent[j].float().cpu().numpy()

            fit_psnr[u, i] = psnr_from_images(pixels[j], images[j])

            valid[u, i] = True

        image_records.append({"id": image_id, "group": "all",

                              "source": source.relative_to(root).as_posix(),

                              "targets": {e: t.relative_to(root).as_posix() for e, t in targets.items()},

                              "size": list(original.size)})

        if (i + 1) % 100 == 0 or i + 1 == len(rows):

            print(f"编码 A-E 专家 z：{i + 1}/{len(rows)} 张原图，已完成 {int(valid.sum())} 个图片对", flush=True)

    means, stds = np.full((5, 1, 3), np.nan, dtype=np.float32), np.full((5, 1, 3), np.nan, dtype=np.float32)

    for u in range(5):

        if valid[u].any():

            means[u, 0] = z[u, valid[u]].mean(0)

            stds[u, 0] = z[u, valid[u]].std(0)

    metadata = {**context, "checkpoint": str(resolve(PHASE1_CHECKPOINT)), "raw_dir": str(root),

                "num_users": 5, "images_per_user": len(rows),

                "valid_images_per_user": {e: int(valid[u].sum()) for u, e in enumerate("ABCDE")},

                "z_dimension": 3, "generation_method": "phase1.encode(original, expert), eval mode",

                "latent_file": "user_latents.npz",

                "latent_axes": {"z": ["user", "image", "dimension"],

                                "zmean_zstd": ["user", "group", "dimension"]},

                "note": "all 是统计组，不影响训练划分；缺失专家图片的位置为 NaN，训练仅使用 valid=True。",

                "users": [{"id": e} for e in "ABCDE"], "images": image_records,

                "created_at_utc": datetime.now(timezone.utc).isoformat()}

    temp_npz = root / ".expert_latents.tmp.npz"

    temp_json = root / ".expert_parameters.tmp.json"

    try:

        np.savez_compressed(temp_npz, users=np.asarray(list("ABCDE")),

                            ids=np.asarray([r[0] for r in rows]), groups=np.asarray(["all"]),

                            image_groups=np.asarray(["all"] * len(rows)),

                            zmean=means, zstd=stds, z=z, fit_psnr=fit_psnr, valid=valid)

        temp_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        # 元数据绑定 npz 内容，防止中断或并发写入造成旧编号与新向量错配。
        metadata["latent_sha256"] = file_sha256(temp_npz)

        temp_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        os.replace(temp_npz, npz_path)

        os.replace(temp_json, metadata_path)

    finally:

        for path in (temp_npz, temp_json):

            if path.exists():

                path.unlink()

    print(f"已保存专家 z：{npz_path}，shape={z.shape}，有效图片对={int(valid.sum())}", flush=True)

    return npz_path, metadata_path





def load_expert_records(expert, phase1, device):

    if expert not in ("A", "B", "C", "D", "E"):

        raise ValueError(f"无效专家：{expert}")

    if not (0 < TRAIN_END < VAL_END):

        raise ValueError("训练/验证编号边界不合法")

    npz_path, metadata_path = ensure_expert_cache(phase1, device)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata["latent_sha256"] != file_sha256(npz_path):

        raise ValueError("专家 z 缓存与元数据不一致，请重新生成")

    with np.load(npz_path, allow_pickle=False) as data:

        users, ids, zs, valid = data["users"].tolist(), data["ids"].tolist(), data["z"], data["valid"]

        u = users.index(expert)

        train_records, val_records, skipped = [], [], 0

        for i, (image_id, row) in enumerate(zip(ids, metadata["images"])):

            if image_id != row["id"]:

                raise ValueError("专家 z 图片编号与元数据不一致")

            number = int(image_id)

            if not 0 <= number < VAL_END:

                continue

            if not valid[u, i]:

                skipped += 1

                continue

            source = resolve(RAW_DIR) / row["source"]

            target = resolve(RAW_DIR) / row["targets"][expert]

            record = (source, target, zs[u, i].copy(), image_id)

            (train_records if number < TRAIN_END else val_records).append(record)

    if not train_records or not val_records:

        raise ValueError(f"专家 {expert} 训练/验证集为空：{len(train_records)}/{len(val_records)}")

    print(f"专家 {expert}：训练 {len(train_records)}、验证 {len(val_records)}，缺失专家目标跳过 {skipped}", flush=True)

    return train_records, val_records





class ImageDataset(Dataset):

    def __init__(self, records): self.records = records

    def __len__(self): return len(self.records)

    def __getitem__(self, i): return image_input(rgb_image(self.records[i][0])), i





class PairwiseZDataset(Dataset):
    """返回原图和三维真实 z。"""
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        source, _, z, _ = self.records[i]
        return image_input(rgb_image(source)), torch.as_tensor(z, dtype=torch.float32)


class ZAwareBatchSampler(Sampler):
    """无正负标签的 batch 采样；每轮随机遍历 anchor，批内掺入其近邻与随机样本。

    允许部分邻居跨批出现，确保每个 epoch 至少覆盖每个训练 id 一次。
    """
    def __init__(self, records, batch_size, top_n, neighbor_fraction, seed):
        if batch_size < 2 or not 0 <= neighbor_fraction <= 1:
            raise ValueError("batch_size >= 2 且邻居占比应在 0..1")
        z = np.stack([r[2] for r in records]).astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError("训练 z 存在非有限值")
        self.size, self.batch_size = len(records), min(batch_size, len(records))
        self.neighbor_num = min(max(int(top_n), 1), max(len(records)-1, 1))
        self.neighbor_fraction = neighbor_fraction
        self.seed = seed
        d = np.square(z[:, None, :] - z[None, :, :]).sum(2)
        np.fill_diagonal(d, np.inf)
        self.neighbors = np.argsort(d, axis=1, kind="stable")[:, :self.neighbor_num]
        self.epoch = 0

    def __len__(self):
        anchor_count = max(1, self.batch_size - round(self.batch_size * self.neighbor_fraction))
        return math.ceil(self.size / anchor_count)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(self.size)
        # 随机打散后每个 anchor 至少作为基础位置出现一次；其他样本补充进批中。
        anchor_count = max(1, self.batch_size - round(self.batch_size * self.neighbor_fraction))
        batches = []
        for start in range(0, len(order), anchor_count):
            anchors = order[start:start+anchor_count].tolist()
            batch = list(anchors)
            used = set(batch)
            for a in anchors:
                candidates = rng.permutation(self.neighbors[a])
                for v in candidates:
                    j = int(v)
                    if j not in used:
                        batch.append(j); used.add(j)
                        break
                if len(batch) >= self.batch_size:
                    break
            if len(batch) < self.batch_size:
                remaining = [int(v) for v in rng.permutation(self.size) if int(v) not in used]
                batch.extend(remaining[:self.batch_size-len(batch)])
            if len(batch) >= 2:
                batches.append(batch)
        # 最后有一个单图 anchor 时，也补近邻变为有效 batch。
        for b in batches:
            yield b


class ImageRetrievalEncoder(nn.Module):
    """全量可训练 DINOv2 + 两层 GELU 投影，输出不归一化的距离嵌入。"""
    def __init__(self, backbone, embedding_dim=EMBEDDING_DIM, hidden_dim=PROJECTION_HIDDEN_DIM):
        super().__init__()
        if embedding_dim < 1 or hidden_dim < 1:
            raise ValueError("投影隐藏维数和输出维数必须为正数")
        self.backbone = backbone.requires_grad_(True)
        self.projection = nn.Sequential(nn.Linear(768, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim))
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)
        self.model_config = dict(architecture="dinov2_vitb14_weighted_z_distance_full_mlp_v2",
                                 embedding_dim=embedding_dim, projection_hidden_dim=hidden_dim,
                                 full_finetune=True, train_last_blocks=12, image_size=IMAGE_SIZE)

    def forward(self, image):
        features = self.backbone(image)
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        return self.projection(features)


def load_pretrained_dinov2():
    print(f"加载官方 DINOv2 模型 {DINO_MODEL}（首次运行可能下载权重）...", flush=True)
    backbone = torch.hub.load(DINO_REPO, DINO_MODEL, pretrained=True,
                              source=DINO_HUB_SOURCE, trust_repo=True)
    if not hasattr(backbone, "blocks") or len(backbone.blocks) != 12:
        raise RuntimeError("DINOv2 ViT-B/14 网络结构不符合预期（预期 12 个 blocks）")
    net = ImageRetrievalEncoder(backbone)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    if frozen: raise RuntimeError(f"全量微调模式仍有 {frozen} 个冻结参数")
    print(f"DINOv2 全量解冻，可训练参数 {trainable/1e6:.2f} M，冻结参数 {frozen}；"
          f"投影 768 -> {PROJECTION_HIDDEN_DIM} -> GELU -> {EMBEDDING_DIM}", flush=True)
    return net


def load_phase1(checkpoint_path, device):

    from phase1.model import Model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if int(checkpoint["dim"]) != 3: raise ValueError("phase1 必须使用三维 z")

    model = Model(dim=3, width=int(checkpoint["width"]), lut_size=int(checkpoint["lut_size"]))

    model.load_state_dict(checkpoint["model"], strict=True)

    return model.to(device).eval().requires_grad_(False)





def psnr_from_images(x, y):

    if x.shape != y.shape: raise ValueError(f"图像尺寸不一致：{x.shape}, {y.shape}")

    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2) / 255. ** 2

    return float("inf") if mse == 0 else -10. * math.log10(mse)





def baseline_psnr(records):

    return float(np.mean([psnr_from_images(*read_pair(src, tgt)) for src, tgt, _, _ in records]))





def distance_matching_loss(embeddings, z):
    """全部不同图片对的连续距离加权 MSE；对近邻更关注、保留远距离监督。"""
    if embeddings.shape[0] < 2:
        raise ValueError("训练 batch 至少需要两张不同图片")
    predicted = torch.pdist(embeddings.float(), p=2)
    target = torch.pdist(z.float(), p=2)
    weight = LOSS_WEIGHT_FLOOR + (1.0 - LOSS_WEIGHT_FLOOR) * torch.exp(-target / LOSS_TAU)
    error = predicted - target
    loss = (weight * error.square()).sum() / weight.sum().clamp_min(1e-8)
    mae = error.abs().mean()
    return loss, mae, predicted.mean(), target.mean()


def encode_images(net, loader, device):

    net.eval()

    features = torch.empty(len(loader.dataset), net.model_config['embedding_dim'], dtype=torch.float32)

    for images, indices in loader:

        encoded = net(images.to(device, non_blocking=True)).float().cpu()

        if not torch.isfinite(encoded).all():

            raise ValueError("图片编码出现无效值")

        features[indices] = encoded

    return features





def retrieve_indices(queries, gallery, k=TOPK):
    """查询编码距离最近的 k 个训练原图；只在 embedding 空间搜索。"""
    if len(gallery) == 0:
        raise ValueError("检索库为空")
    ids, distances = [], []
    k = min(int(k), len(gallery))
    for chunk in queries.split(256):
        distance = torch.cdist(chunk.float(), gallery.float(), p=2)
        best_d, best_i = distance.topk(k, dim=1, largest=False, sorted=True)
        ids.append(best_i)
        distances.append(best_d)
    return torch.cat(ids), torch.cat(distances)


def aggregate_neighbor_z(indices, distances, train_z, temperature=TOPK_TEMPERATURE):
    """仅通过编码距离加权融合训练库中的 z；不能访问查询的真实 z。"""
    weights = torch.softmax(-distances.float() / temperature, dim=1)
    z_tensor = torch.from_numpy(train_z).float()
    averaged = (z_tensor[indices] * weights.unsqueeze(-1)).sum(dim=1)
    return averaged.numpy(), weights.numpy()


def save_val_visual(path, original, retouched, target, reference_original, reference_edited,

                    image_id, epoch, neighbor_id, psnr):

    """第一行：原图/TopK z 融合修图/目标；第二行：Top1 命中参考原图/真实修图。"""

    h = max(original.shape[0], reference_original.shape[0])

    w = max(original.shape[1], reference_original.shape[1])

    gap, header, footer = 8, 28, 42

    row_height = h + header

    canvas = Image.new('RGB', (3 * w + 2 * gap, 2 * row_height + gap + footer), (245, 245, 245))

    draw = ImageDraw.Draw(canvas)

    try:

        font = ImageFont.truetype('arial.ttf', 14)

    except OSError:

        font = ImageFont.load_default()

    rows = (((original, 'Original'), (retouched, 'TopK blended retouch'), (target, 'Target')),

            ((reference_original, 'Reference original'), (reference_edited, 'Reference retouched')))

    for row, panels in enumerate(rows):

        y = row * (row_height + gap)

        for col, (pixels, title) in enumerate(panels):

            x = col * (w + gap)

            ph, pw = pixels.shape[:2]

            canvas.paste(Image.fromarray(pixels), (x + (w - pw) // 2, y + header + (h - ph) // 2))

            box = draw.textbbox((0, 0), title, font=font)

            draw.text((x + max(0, (w - box[2]) // 2), y + 5), title, font=font, fill=(30, 30, 30))

    bottom = 2 * row_height + gap

    draw.text((5, bottom + 3), f'Image {image_id} | Epoch {epoch} | Top1 reference {neighbor_id}', font=font, fill=(30, 30, 30))

    draw.text((5, bottom + 22), f'Retouched vs target PSNR: {psnr:.3f} dB', font=font, fill=(30, 30, 30))

    canvas.save(path)





@torch.inference_mode()

def rendered_psnrs(decoder, records, z_values, device, visual_dir=None, epoch=None, neighbor_ids=None, reference_records=None):

    if len(records) != len(z_values): raise ValueError("图片与检索 z 数量不一致")

    if visual_dir is not None:

        visual_dir = Path(visual_dir)

        visual_dir.mkdir(parents=True, exist_ok=True)

        if neighbor_ids is None or len(neighbor_ids) != len(records): raise ValueError("可视化需要每张评估图的检索邻居编号")

        if reference_records is None or len(reference_records) != len(records): raise ValueError("可视化需要每张评估图命中的训练记录")

        if [r[3] for r in reference_records] != list(neighbor_ids): raise ValueError("参考图片记录与邻居编号不一致")

    scores = []

    for i, ((source, target, _, image_id), z) in enumerate(zip(records, z_values)):

        original, edited = read_pair(source, target)

        image = torch.from_numpy(original.copy()).permute(2, 0, 1)[None].to(device).float() / 255.

        latent = torch.as_tensor(z, dtype=torch.float32, device=device).reshape(1, 3)

        output = decoder.decode(image, latent)[0]

        if not torch.isfinite(output).all(): raise ValueError(f"phase1 重建无效：{image_id}")

        pixels = (output.clamp(0, 1) * 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()

        score = psnr_from_images(pixels, edited)

        scores.append(score)

        if visual_dir is not None:

            ref_source, ref_target, _, _ = reference_records[i]

            reference_original, reference_edited = read_pair(ref_source, ref_target)

            save_val_visual(visual_dir/f'{image_id}.png', original, pixels, edited, reference_original,

                            reference_edited, image_id, epoch, neighbor_ids[i], score)

    return np.asarray(scores, np.float64)





def load_category_lookup():

    expert = expert_code()

    if expert:

        from user import ensure_expert_groups

        manifest = ensure_expert_groups(resolve(RAW_DIR), resolve(EXPERT_GROUP_DIR), experts=expert)[expert]

    else:

        manifest = resolve(USER_DIR).parent/'user_parameters.json'

    metadata = json.loads(manifest.read_text(encoding='utf-8'))

    return {int(row['id']): str(row['group']) for row in metadata['images']

            if row.get('group') is not None and str(row['group']).strip() not in ('', 'all')}





@torch.inference_mode()
def evaluate(net, decoder, train_records, val_records, loader, device, epoch=None, category_lookup=None):
    if len(loader.dataset) != len(train_records) + len(val_records):
        raise ValueError("评估编码数据数量错误")
    features = encode_images(net, loader, device)
    bank = features[:len(train_records)]
    queries = features[len(train_records):]
    bank_z = np.stack([r[2] for r in train_records]).astype(np.float32)
    true_z = np.stack([r[2] for r in val_records]).astype(np.float32)
    indices, distances = retrieve_indices(queries, bank, TOPK)
    matched = indices[:, 0]
    top1_z = bank_z[matched.numpy()]
    top5_z, weights = aggregate_neighbor_z(indices, distances, bank_z)
    visual_dir = resolve(VISUAL_DIR)/resolve(expert_code() or USER_DIR).name if SAVE_VAL_VISUAL and epoch is not None else None
    top1_scores = rendered_psnrs(decoder, val_records, top1_z, device)
    top5_scores = rendered_psnrs(decoder, val_records, top5_z, device, visual_dir, epoch,
                                [train_records[int(j)][3] for j in matched],
                                [train_records[int(j)] for j in matched])
    if visual_dir is not None:
        print(f"val TopK 两行对比图已保存：{visual_dir}", flush=True)
    mse1 = np.square(top1_z.astype(np.float64) - true_z).mean(axis=1)
    mse5 = np.square(top5_z.astype(np.float64) - true_z).mean(axis=1)
    rows = []
    category_correct = category_total = 0
    for i, (_, _, _, image_id) in enumerate(val_records):
        j = int(matched[i]); neighbor_ids = [train_records[int(v)][3] for v in indices[i]]
        row = dict(query_id=image_id, neighbor_id=train_records[j][3],
                   embedding_distance=float(distances[i, 0]),
                   topk_neighbor_ids=";".join(neighbor_ids),
                   topk_distances=";".join(f"{float(v):.6g}" for v in distances[i]),
                   topk_weights=";".join(f"{float(v):.6g}" for v in weights[i]),
                   top1_z1=float(top1_z[i, 0]), top1_z2=float(top1_z[i, 1]), top1_z3=float(top1_z[i, 2]),
                   top5_z1=float(top5_z[i, 0]), top5_z2=float(top5_z[i, 1]), top5_z3=float(top5_z[i, 2]),
                   top1_z_mse=float(mse1[i]), top5_z_mse=float(mse5[i]),
                   top1_psnr_db=float(top1_scores[i]), top5_psnr_db=float(top5_scores[i]))
        if category_lookup is not None:
            query_group = category_lookup.get(int(image_id))
            neighbor_group = category_lookup.get(int(train_records[j][3]))
            correct = query_group == neighbor_group if query_group is not None and neighbor_group is not None else None
            row.update(query_group=query_group, neighbor_group=neighbor_group, category_correct=correct)
            if correct is not None:
                category_total += 1
                category_correct += int(correct)
        rows.append(row)
    gallery = dict(ids=[r[3] for r in train_records], sources=[str(r[0]) for r in train_records],
                   embeddings=bank.clone(), z=torch.from_numpy(bank_z))
    return dict(val_psnr=float(top5_scores.mean()), val_z_mse=float(mse5.mean()),
                top1_psnr=float(top1_scores.mean()), top5_psnr=float(top5_scores.mean()),
                top1_z_mse=float(mse1.mean()), top5_z_mse=float(mse5.mean()),
                gallery_feature_std=float(bank.std(dim=0, unbiased=False).mean()),
                unique_neighbors=int(matched.unique().numel()), gallery=gallery, rows=rows,
                val_category_correct=category_correct, val_category_total=category_total,
                val_category_accuracy=category_correct/category_total if category_total else None)


def save_csv(path, rows):

    with path.open('w', newline='', encoding='utf-8-sig') as f:

        writer = csv.DictWriter(f, fieldnames=list(rows[0]))

        writer.writeheader(); writer.writerows(rows)





def save_checkpoint(path, net, optimizer, scheduler, epoch, result, phase1_hash, scaler):
    torch.save(dict(epoch=epoch, model=net.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict() if scaler is not None else None,
                    val_z_mse=result['top5_z_mse'], val_psnr=result['top5_psnr'],
                    val_top1_z_mse=result['top1_z_mse'], val_top1_psnr=result['top1_psnr'],
                    val_top5_z_mse=result['top5_z_mse'], val_top5_psnr=result['top5_psnr'],
                    backbone=DINO_MODEL, model_config=net.model_config,
                    image_size=IMAGE_SIZE, user_dir=str(resolve(expert_code() or USER_DIR)),
                    pretrained_source=f"{DINO_REPO}/{DINO_MODEL}", phase1_checkpoint_sha256=phase1_hash,
                    objective="z_aware_weighted_pairwise_euclidean_distance_mse",
                    loss_tau=LOSS_TAU, loss_weight_floor=LOSS_WEIGHT_FLOOR,
                    topk=TOPK, topk_temperature=TOPK_TEMPERATURE,
                    gallery=result['gallery'], train_end=TRAIN_END, val_end=VAL_END), path)


def reference_baseline_zs(train_records, val_records, category_lookup):

    """两种使用真实标签的诊断参照；检索候选和组均值都仅来自训练集。"""

    bank = np.stack([r[2] for r in train_records]).astype(np.float64)

    queries = np.stack([r[2] for r in val_records]).astype(np.float64)

    nearest = np.empty(len(queries), np.int64)

    for start in range(0, len(queries), 256):

        distance = np.square(queries[start:start+256, None] - bank[None]).sum(2)

        nearest[start:start+len(distance)] = distance.argmin(1)

    grouped = {}

    for record, z in zip(train_records, bank):

        group = category_lookup.get(int(record[3])) if category_lookup is not None else None

        if group is not None: grouped.setdefault(group, []).append(z)

    means = {g: np.mean(values, axis=0) for g, values in grouped.items()}

    fallback = bank.mean(0); missing = 0; group_z = []

    for record in val_records:

        group = category_lookup.get(int(record[3])) if category_lookup is not None else None

        if group in means: group_z.append(means[group])

        else: group_z.append(fallback); missing += 1

    return bank[nearest].astype(np.float32), np.asarray(group_z, np.float32), missing





def compute_startup_baselines(decoder, train_records, val_records, category_lookup, device):

    original = baseline_psnr(val_records)

    fitted = float(rendered_psnrs(decoder, val_records, [r[2] for r in val_records], device).mean())

    nearest_z, mean_z, fallback_count = reference_baseline_zs(train_records, val_records, category_lookup)

    nearest = float(rendered_psnrs(decoder, val_records, nearest_z, device).mean())

    mean = float(rendered_psnrs(decoder, val_records, mean_z, device).mean())

    print(f"开局基线（评估 {len(val_records)} 张；仅供参考，不参与编码检索）：", flush=True)

    print(f"  原图 vs 修图：{original:.4f} dB", flush=True)

    print(f"  逐图拟合 z 重建 vs 修图：{fitted:.4f} dB", flush=True)

    print(f"  真实 z 最近训练图重建 vs 修图：{nearest:.4f} dB", flush=True)

    print(f"  真实分组的训练集平均 z 重建 vs 修图：{mean:.4f} dB", flush=True)

    if fallback_count: print(f"  {fallback_count} 张查询缺少对应训练组，组均值基线回退为训练库整体均值", flush=True)

    return original, fitted, nearest, mean





def main():

    if (EPOCHS < 1 or BATCH_SIZE < 2 or EVAL_BATCH_SIZE < 1 or IMAGE_SIZE % 14 != 0
            or LOSS_TAU <= 0 or TOPK_TEMPERATURE <= 0 or not 0 < LOSS_WEIGHT_FLOOR <= 1
            or ACCUMULATION_STEPS < 1 or TOPK < 1 or EMBEDDING_DIM < 1 or PROJECTION_HIDDEN_DIM < 1):
        raise ValueError("训练参数不合法：DINO 输入尺寸必须为 14 的倍数，温度与 batch 参数必须有效")


    device = torch.device(DEVICE)

    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError("CUDA 不可用")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if device.type == 'cuda':

        torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark = False
    phase1_path = resolve(PHASE1_CHECKPOINT)

    phase1_hash = file_sha256(phase1_path)

    decoder = load_phase1(phase1_path, device)

    if file_sha256(phase1_path) != phase1_hash: raise RuntimeError("加载期间 phase1 权重改变，请重新运行")

    expert = expert_code(); user_dir = resolve(expert or USER_DIR)

    if expert:

        train_records, val_records = load_expert_records(expert, decoder, device)

    else:

        # 保证监督 z 与用于渲染的 phase1 属于同一个模型空间。
        from analyze_z import refresh_cache

        import analyze_z

        analyze_z.CHECKPOINT = str(phase1_path)

        metadata = json.loads((user_dir.parent/'user_parameters.json').read_text(encoding='utf-8'))

        checkpoint_info = metadata.get('checkpoint_info', {})

        if metadata.get('checkpoint_sha256') != phase1_hash:

            ckpt = torch.load(phase1_path, map_location='cpu', weights_only=True)

            checkpoint_info = {k: ckpt.get(k) for k in ('dim','width','lut_size','epoch','val_psnr','dataset')}

            del ckpt

        encoder_hash = hashlib.sha256(''.join(file_sha256(ROOT/'phase1'/p) for p in ('model.py','config.py')).encode()).hexdigest()

        refresh_cache(user_dir.parent, decoder, device, phase1_hash, checkpoint_info, encoder_hash)

        train_records, val_records = load_records(user_dir)

    if {int(r[3]) for r in train_records} & {int(r[3]) for r in val_records}:

        raise ValueError("训练与评估存在相同原图编号")

    category_lookup = load_category_lookup()

    if len(train_records) < 2:
        raise ValueError("至少需要两张训练图以构建距离监督")
    train_z = np.stack([r[2] for r in train_records])
    if train_z.shape != (len(train_records), 3) or not np.isfinite(train_z).all():
        raise ValueError("训练 z 必须是有限的三维向量")
    out_dir = user_dir / OUTPUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=device.type=='cuda', persistent_workers=NUM_WORKERS>0)
    sampler = ZAwareBatchSampler(train_records, BATCH_SIZE, Z_NEIGHBOR_TOP_N, Z_NEIGHBOR_FRACTION, SEED)
    train_loader = DataLoader(PairwiseZDataset(train_records), batch_sampler=sampler, **kw)
    eval_loader = DataLoader(ImageDataset(train_records + val_records), batch_size=EVAL_BATCH_SIZE, shuffle=False, **kw)

    net = load_pretrained_dinov2().to(device)
    print(f"用户 {user_dir.name}：训练 {len(train_records)}，验证 {len(val_records)}，"
          f"DINOv2 ViT-B/14 全量微调，MLP 输出 {EMBEDDING_DIM}D，无正负标签", flush=True)


    base, fitted_base, nearest_base, group_mean_base = compute_startup_baselines(

        decoder, train_records, val_records, category_lookup, device)

    params_backbone = [p for p in net.backbone.parameters() if p.requires_grad]
    params_projection = list(net.projection.parameters())
    optimizer = torch.optim.AdamW([
        dict(params=params_backbone, lr=BACKBONE_LR),
        dict(params=params_projection, lr=PROJECTION_LR)], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    use_amp = USE_AMP and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if AMP_DTYPE == 'bfloat16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
    best_psnr = -float('inf')
    columns = ['epoch','train_weighted_distance_mse','train_distance_mae','train_embedding_pair_distance','train_z_pair_distance',
               'top1_z_mse','top5_z_mse','top1_psnr_db','top5_psnr_db',
               'baseline_psnr_db','fitted_z_baseline_psnr_db','nearest_z_baseline_psnr_db',
               'group_mean_z_baseline_psnr_db','unique_neighbors','gallery_feature_std',
               'backbone_lr','projection_lr','val_category_correct','val_category_total','val_category_accuracy']


    with (out_dir/'training.csv').open('w', newline='', encoding='utf-8') as file:

        writer = csv.writer(file); writer.writerow(columns)

        for epoch in range(1, EPOCHS + 1):
            net.train()
            totals = np.zeros(4, dtype=np.float64)
            count = 0
            optimizer.zero_grad(set_to_none=True)
            for step, (images, batch_z) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                batch_z = batch_z.to(device, non_blocking=True)
                if len(images) < 2:
                    continue
                autocast_ctx = torch.autocast('cuda', dtype=amp_dtype) if use_amp else nullcontext()
                with autocast_ctx:
                    outputs = net(images)
                loss, dist_mae, emb_dist, z_dist = distance_matching_loss(outputs, batch_z)
                if not torch.isfinite(loss):
                    raise ValueError("距离监督损失非有限值")
                # 最后一组不足累积步数时也采用正确的归一化
                group_start = (step // ACCUMULATION_STEPS) * ACCUMULATION_STEPS
                group_size = min(ACCUMULATION_STEPS, len(train_loader) - group_start)
                scaler.scale(loss / group_size).backward()
                if (step + 1) % ACCUMULATION_STEPS == 0 or step + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_((p for p in net.parameters() if p.requires_grad), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                totals += np.asarray([loss.item(), dist_mae.item(), emb_dist.item(), z_dist.item()]) * len(images)
                count += len(images)
            if count == 0:
                raise RuntimeError("训练集没有有效的成对样本")
            train_loss, distance_mae, embedding_pair_distance, z_pair_distance = totals / count
            result = evaluate(net, decoder, train_records, val_records, eval_loader, device,
                              epoch=epoch, category_lookup=category_lookup)
            lr_backbone, lr_projection = (g['lr'] for g in optimizer.param_groups)
            correct, total = result['val_category_correct'], result['val_category_total']
            log_row = [epoch, train_loss, distance_mae, embedding_pair_distance, z_pair_distance,
                       result['top1_z_mse'], result['top5_z_mse'], result['top1_psnr'], result['top5_psnr'],
                       base, fitted_base, nearest_base, group_mean_base, result['unique_neighbors'],
                       result['gallery_feature_std'], lr_backbone, lr_projection,
                       correct, total, result['val_category_accuracy']]
            writer.writerow(log_row)
            file.flush()
            category_text = f' | 类别 {correct}/{total} ({correct/total:.2%})' if total else ''
            print(f"Epoch {epoch:03d}/{EPOCHS} | weighted MSE {train_loss:.6f} | MAE {distance_mae:.4f} "
                  f"| Top1 PSNR {result['top1_psnr']:.4f} dB | Top5 PSNR {result['top5_psnr']:.4f} dB "
                  f"| Top1/5 z MSE {result['top1_z_mse']:.6f}/{result['top5_z_mse']:.6f}"
                  f" | unique {result['unique_neighbors']}{category_text}", flush=True)
            scheduler.step()
            if SAVE_EVERY_EPOCH:
                save_checkpoint(out_dir/'latest.pt', net, optimizer, scheduler, epoch, result, phase1_hash, scaler)
                save_csv(out_dir/'latest_retrieval.csv', result['rows'])
            score = result['top5_psnr'] if BEST_METRIC == 'top5_psnr' else result['top1_psnr']
            if score > best_psnr:
                best_psnr = score
                save_checkpoint(out_dir/'best.pt', net, optimizer, scheduler, epoch, result, phase1_hash, scaler)
                save_csv(out_dir/'best_retrieval.csv', result['rows'])
                print(f"更新最佳模型：{out_dir/'best.pt'}", flush=True)


    print(f"完成：最佳 {BEST_METRIC}={best_psnr:.4f} dB，输出 {out_dir}", flush=True)





if __name__ == '__main__': main()
