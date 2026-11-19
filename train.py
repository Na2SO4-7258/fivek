# ========================= 超参数 =========================
USER_DIR = r"data\semantic\user1"   # 路径=模拟用户；"A"/"B"/"C"/"D"/"E"=单个真实专家data\semantic\user2"
RAW_DIR = r"data\raw"               # 专家模式读取 编号/O.png 和 编号/专家.png
PHASE1_CHECKPOINT = r"phase1\out\model\best.pt"
PRETRAINED_PATH = r"model\convnext_tiny_imagenet1k_v1.pth"  # 与之前 CLIP 同在 model/ 下
TRAIN_END = 1500                # 按文件名数字编号训练 [0, 1500)
VAL_END = 2000                  # 按文件名数字编号验证 [1500, 2000)
EPOCHS = 100
BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4                  # Windows 如遇多进程问题改为 0
SEED = 42
DEVICE = "cuda"
SAVE_EVERY_EPOCH = True
# ==========================================================

"""运行：python train.py
依赖：pip install torch torchvision pillow numpy

输入：data/semantic/类别/编号.png 原图；data/semantic/user1/编号.png 目标修图。
监督：模拟用户读取 data/semantic/user_latents.npz 中缓存的真实 3 维 z。
USER_DIR="A"（或 B/C/D/E）时，从 RAW_DIR 读取 O.png 与对应专家.png，
用冻结的 phase1 编码得到三维 z；自动缓存到 RAW_DIR/user_latents.npz 和 user_parameters.json。
缓存结构与模拟用户相同：users、ids、groups、image_groups、zmean、zstd、z、fit_psnr。
真实专家统一以 all 为统计组，缺失图片对由 valid 标记；不使用旧 *z.npy。
专家修图尺寸不同时按原图尺寸用 LANCZOS 对齐，编码与 PSNR 采用同一对齐方式。
专家模式输出：项目 A/convnext_z/（B–E 同理），文件名和记录格式与模拟用户相同。
按图片文件名数字编号划分训练/验证，严格按 id 对应 z，不使用缓存行序号划分。
模型：ImageNet 预训练 ConvNeXt-Tiny + 3 维回归头，端到端微调。
预训练权重保存到项目 model/ 目录；仅当文件不存在时自动下载。
验证：预测 z 经原 phase1.decode(O,z) 生成 RGB，逐图与用户修图计算 PSNR。
启动时输出原图基线和拟合 z 重建基线的 val PSNR；每个 epoch 同时记录两项基线。
训练结果存所选用户文件夹下 convnext_z/（best.pt、latest.pt、training.csv）。
"""

import csv
import hashlib
import os
import json
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

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


class ZDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        source, _, z, _ = self.records[idx]
        x = image_input(rgb_image(source))
        return x, torch.from_numpy(z)


def load_pretrained_convnext():
    """本地文件优先；没有则下载到项目 model/ 目录，之后不重复下载。"""
    weight_path = resolve(PRETRAINED_PATH)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    if not weight_path.is_file():
        print(f"首次下载 ConvNeXt-Tiny 预训练权重到：{weight_path}", flush=True)
        # load_state_dict_from_url 支持自定义 model_dir 并下载到临时文件后完成保存。
        from torch.hub import load_state_dict_from_url
        state = load_state_dict_from_url(weights.url, model_dir=str(weight_path.parent),
                                        file_name=weight_path.name, map_location="cpu", progress=True,
                                        check_hash=False, weights_only=True)
    else:
        print(f"加载已缓存的 ConvNeXt-Tiny 预训练权重：{weight_path}", flush=True)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)

    model = convnext_tiny(weights=None)
    model.load_state_dict(state, strict=True)  # 先载入 ImageNet 分类模型的完整预训练权重
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 3)  # 换成 3 维 z 回归头
    nn.init.normal_(model.classifier[2].weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.classifier[2].bias)
    return model


def load_phase1(checkpoint_path, device):
    from phase1.model import Model
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 phase1 权重：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["dim"]) != 3:
        raise ValueError("phase1 模型必须使用 3 维 z")
    model = Model(dim=3, width=int(checkpoint["width"]), lut_size=int(checkpoint["lut_size"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def psnr_from_images(x, y):
    if x.shape != y.shape:
        raise ValueError(f"原图和用户修图尺寸不一致：{x.shape}, {y.shape}")
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2) / 255.0 ** 2
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def read_pixels(path):
    return np.asarray(rgb_image(path), dtype=np.uint8)


def baseline_psnr(records):
    return float(np.mean([psnr_from_images(*read_pair(src, tgt))
                          for src, tgt, _, _ in records]))


@torch.inference_mode()
def evaluate(net, decoder, val_records, device):
    """net=None 时直接解码缓存的拟合 z，计算相同验证集上的重建基线。"""
    if net is not None:
        net.eval()
    z_squared_error = 0.0
    psnrs = []
    for source, target, z_true, image_id in val_records:
        raw, target_pixels = read_pair(source, target)
        if net is None:
            predicted_z = torch.as_tensor(z_true, dtype=torch.float32, device=device).reshape(1, 3)
        else:
            small = image_input(Image.fromarray(raw)).unsqueeze(0).to(device)
            predicted_z = net(small)
        z_squared_error += float(((predicted_z.cpu().numpy()[0] - z_true) ** 2).sum())

        full = (torch.from_numpy(raw.copy()).permute(2, 0, 1).unsqueeze(0)
                .to(device, dtype=torch.float32) / 255.0)
        output = decoder.decode(full, predicted_z)[0]
        if not torch.isfinite(output).all():
            raise ValueError(f"phase1 解码输出无效：{image_id}")
        predicted_pixels = (output.clamp(0, 1) * 255).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        psnrs.append(psnr_from_images(predicted_pixels, target_pixels))
    return z_squared_error / (len(val_records) * 3), float(np.mean(psnrs))


def save_checkpoint(path, net, optimizer, scheduler, epoch, mse, psnr):
    torch.save({"epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "val_z_mse": mse, "val_psnr": psnr,
                "backbone": "torchvision_convnext_tiny_imagenet1k_v1", "image_size": IMAGE_SIZE,
                "user_dir": str(resolve(expert_code() or USER_DIR)), "pretrained_path": str(resolve(PRETRAINED_PATH))}, path)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("此程序要求 CUDA，请安装 CUDA 版 PyTorch 并检查显卡驱动")
    if EPOCHS < 1 or BATCH_SIZE < 1 or IMAGE_SIZE < 32:
        raise ValueError("EPOCHS/BATCH_SIZE 至少为 1，IMAGE_SIZE 至少为 32")
    device = torch.device(DEVICE)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

    expert = expert_code()
    user_dir = resolve(expert or USER_DIR)
    decoder = None
    if expert:
        decoder = load_phase1(resolve(PHASE1_CHECKPOINT), device)
        train_records, val_records = load_expert_records(expert, decoder, device)
    else:
        train_records, val_records = load_records(user_dir)
    out_dir = user_dir / "convnext_z"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"用户：{user_dir.name} | CUDA：{torch.cuda.get_device_name(device)}", flush=True)
    print(f"训练：[0,{TRAIN_END}) {len(train_records)} 张 | 验证：[{TRAIN_END},{VAL_END}) {len(val_records)} 张", flush=True)
    base = baseline_psnr(val_records)
    print(f"训练前：原图 vs 修图 val PSNR = {base:.4f} dB", flush=True)

    train_loader = DataLoader(ZDataset(train_records), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=NUM_WORKERS > 0)
    net = load_pretrained_convnext().to(device)
    if decoder is None:
        decoder = load_phase1(resolve(PHASE1_CHECKPOINT), device)
    _, fitted_base = evaluate(None, decoder, val_records, device)
    print(f"训练前：拟合 z 重建 vs 修图 val PSNR = {fitted_base:.4f} dB", flush=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    best_psnr = -float("inf")
    log_path = out_dir / "training.csv"

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_z_mse", "val_z_mse", "val_psnr_db", "baseline_psnr_db", "lr", "fitted_z_baseline_psnr_db"])
        for epoch in range(1, EPOCHS + 1):
            net.train()
            total_loss = 0.0
            total_items = 0
            for x, z in train_loader:
                x = x.to(device, non_blocking=True)
                z = z.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                predicted = net(x)
                loss = loss_fn(predicted, z)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x)
                total_items += len(x)

            train_mse = total_loss / total_items
            val_mse, val_psnr = evaluate(net, decoder, val_records, device)
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, train_mse, val_mse, val_psnr, base, lr, fitted_base])
            file.flush()
            print(f"Epoch {epoch:03d}/{EPOCHS} | train z MSE {train_mse:.6f} | "
                  f"val z MSE {val_mse:.6f} | 输出图 vs 修图 val PSNR {val_psnr:.4f} dB "
                  f"| 原图基线 {base:.4f} dB | 拟合 z 基线 {fitted_base:.4f} dB", flush=True)

            scheduler.step()
            if SAVE_EVERY_EPOCH:
                save_checkpoint(out_dir / "latest.pt", net, optimizer, scheduler, epoch, val_mse, val_psnr)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(out_dir / "best.pt", net, optimizer, scheduler, epoch, val_mse, val_psnr)
                print(f"  更新最佳权重：{out_dir / 'best.pt'}", flush=True)

    print(f"完成。最佳 val PSNR = {best_psnr:.4f} dB，结果保存在 {out_dir}", flush=True)


if __name__ == "__main__":
    main()
