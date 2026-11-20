# ========================= 超参数 =========================
# 无需设置分组模式/目录：直接运行时自动跟随 group.py 最近成功运行的分组，每套分组生成一个用户。
RAW_DIR = "data/raw"                         # 仅 --experts 功能：O.png、A-E 修图及三维 z 缓存
EXPERT_GROUP_DIR = "data/expert_groups"       # 仅 --experts 功能：输出 A/1..6、B/1..6 ...
EXPERT_GROUPS = 6                            # 每个专家独立的 z 聚类数量；不同专家同编号不代表同一类
CHECKPOINT = r"phase1\out\model\best.pt"   # phase1 权重（含 encoder + decoder）
NUM_USERS = 1                                # 每套分组固定只生成 user1；模式 3 有 random1..3 三套，故总共三个用户
SEED = 42
DEVICE = "auto"                             # auto / cuda / cuda:0 / cpu
OVERWRITE = 1                            # True：备份旧 user* 和元数据后重新生成

# 复杂修图偏好：每个用户×类别固定一组参数；每张图再加少量扰动
EXPOSURE_RANGE = (-0.40, 0.40)              # EV
CONTRAST_RANGE = (0.7, 1.5)
SATURATION_RANGE = (0.5, 1.60)
GAMMA_RANGE = (0.6, 1.4)
SHADOW_RANGE = (-0.08, 0.20)                # 阴影提亮/压暗
HIGHLIGHT_RANGE = (0.00, 0.28)              # 高光压缩强度
CHANNEL_GAIN_RANGE = (0.88, 1.14)           # RGB 通道增益
WARMTH_RANGE = (-0.18, 0.18)                # 冷暖色偏移
VIBRANCE_RANGE = (0.00, 0.35)               # 额外饱和增强
VIGNETTE_RANGE = (0.00, 0.16)               # 暗角强度
DETAIL_RANGE = (0.90, 1.10)                 # 轻微局部对比
NOISE_SCALE = 0.03                          # 单图扰动占各参数跨度的比例
# ==========================================================

"""运行：python group.py，然后 python user.py。依赖 torch、numpy、Pillow。
分组方式只在 group.py 选择：1 语义→data/semantic，2 色彩→data/color，3 三个随机网络→data/random1..3。
user.py 自动读取最近成功分组的目录和清单；模式 1/2 在一个目录生成 user1，模式 3 在三个目录各生成一个 user1。
命令行 --mode/--output 覆盖也会自动同步；只修改 group.py 的 MODE 而未运行，不会混用两套结果。
保留独立专家分组功能：python user.py --experts，读取/更新原位置专家 z，A-E 各聚六组（需 scikit-learn）。

流程：
1) 对每个用户×类别采样一组稳定的高维修图偏好；对每张原图应用更复杂的图像编辑，生成目标修图。
2) 直接使用 phase1 的 encoder 一次前向传播 model.encode(O, T) 得到三维 z。
3) 使用 phase1 的 decoder 重建并计算 z 拟合 PSNR：decode(O, z) vs T。
4) 保存 user1/编号.png、user_parameters.json、user_latents.npz。

与旧版保持兼容：
- user_latents.npz 仍保存 users、ids、groups、image_groups、zmean、zstd、z。
- 额外增加 fit_psnr，记录每个用户对每张图的 z 拟合 PSNR。
- user_parameters.json 中记录复杂修图偏好，以及每个用户/类别的拟合 PSNR 汇总。
"""

import argparse
import hashlib
import json
import math
import re
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
import group as grouping


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
META_FILES = ("user_parameters.json", "user_latents.npz")


def resolve(path):
    p = Path(path)
    return (p if p.is_absolute() else ROOT / p).resolve()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rgb_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def collect_images(group_dir):
    if not group_dir.is_dir():
        raise FileNotFoundError(f"分组目录不存在：{group_dir}")
    folders = sorted((p for p in group_dir.iterdir() if p.is_dir() and p.name.isdecimal()),
                     key=lambda p: (int(p.name), p.name))
    if not folders:
        raise ValueError("分组目录内需要 1、2、3 ... 这样的数字类别目录")
    rows, seen = [], set()
    for folder in folders:
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_id = path.stem
            if image_id.casefold() in seen:
                raise ValueError(f"样本编号重复，输出会重名：{path}")
            seen.add(image_id.casefold())
            rows.append({"id": image_id, "group": folder.name, "path": path})
    if not rows:
        raise ValueError(f"分组目录内没有可处理的图片：{group_dir}")
    return [p.name for p in folders], rows


def collect_group(root, metadata, info):
    """核对一套 group.py 输出的清单、目录和原图副本。"""
    group_ids, rows = collect_images(root)
    expected = {}
    for row in metadata['images']:
        key = str(row['id']).casefold()
        if key in expected: raise ValueError(f'分组清单含重复编号：{key}')
        expected[key] = row
    if len(rows) != metadata['total'] or {r['id'].casefold() for r in rows} != set(expected):
        raise ValueError('分组目录的图片与 group.py 清单不一致，请先重新分组')
    if set(group_ids) != {str(c['id']) for c in metadata['categories']}:
        raise ValueError('分组目录的类别与 group.py 清单不一致，请先重新分组')
    for row in rows:
        entry = expected[row['id'].casefold()]
        if row['group'] != str(entry['group']) or row['path'].resolve() != (root/entry['file']).resolve():
            raise ValueError(f"图片 {row['id']} 的类别/路径与分组清单不一致")
        row['expected_sha256'] = entry.get('source_sha256')
    # 语义类别允许为空；仅对有图片的类别采样偏好、计算 z 统计，避免空组 NaN。
    occupied = {r['group'] for r in rows}
    group_ids = [g for g in group_ids if g in occupied]
    return root, group_ids, rows, info


def collect_current_groups():
    """模式 1/2 返回一套；模式 3 返回 random1、random2、random3 三套。"""
    return [collect_group(*item) for item in grouping.current_groups()]


def rng_for(*keys):
    payload = json.dumps([SEED, *keys], ensure_ascii=False).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"))


def sample_scalar(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def sample_preferences(group_ids, dataset_key):
    users = []
    for user in range(NUM_USERS):
        prefs = {}
        for gid in group_ids:
            rng = rng_for("preferences", dataset_key, user + 1, gid)
            prefs[gid] = {
                "exposure": sample_scalar(rng, EXPOSURE_RANGE),
                "contrast": sample_scalar(rng, CONTRAST_RANGE),
                "saturation": sample_scalar(rng, SATURATION_RANGE),
                "gamma": sample_scalar(rng, GAMMA_RANGE),
                "shadow": sample_scalar(rng, SHADOW_RANGE),
                "highlight": sample_scalar(rng, HIGHLIGHT_RANGE),
                "channel_gains": [sample_scalar(rng, CHANNEL_GAIN_RANGE) for _ in range(3)],
                "warmth": sample_scalar(rng, WARMTH_RANGE),
                "vibrance": sample_scalar(rng, VIBRANCE_RANGE),
                "vignette": sample_scalar(rng, VIGNETTE_RANGE),
                "detail": sample_scalar(rng, DETAIL_RANGE),
            }
        users.append(prefs)
    return users


def perturb_preferences(base, image_id, user_idx, gid, dataset_key):
    rng = rng_for("image-pref", dataset_key, user_idx + 1, gid, image_id)

    def jitter(key, span):
        return float(base[key] + rng.normal(0.0, NOISE_SCALE * span))

    params = {
        "exposure": jitter("exposure", EXPOSURE_RANGE[1] - EXPOSURE_RANGE[0]),
        "contrast": max(0.10, jitter("contrast", CONTRAST_RANGE[1] - CONTRAST_RANGE[0])),
        "saturation": max(0.05, jitter("saturation", SATURATION_RANGE[1] - SATURATION_RANGE[0])),
        "gamma": max(0.10, jitter("gamma", GAMMA_RANGE[1] - GAMMA_RANGE[0])),
        "shadow": jitter("shadow", SHADOW_RANGE[1] - SHADOW_RANGE[0]),
        "highlight": float(np.clip(jitter("highlight", HIGHLIGHT_RANGE[1] - HIGHLIGHT_RANGE[0]), 0.0, 0.95)),
        "channel_gains": [
            float(np.clip(g + rng.normal(0.0, NOISE_SCALE * (CHANNEL_GAIN_RANGE[1] - CHANNEL_GAIN_RANGE[0])), 0.50, 1.80))
            for g in base["channel_gains"]
        ],
        "warmth": jitter("warmth", WARMTH_RANGE[1] - WARMTH_RANGE[0]),
        "vibrance": max(0.0, jitter("vibrance", VIBRANCE_RANGE[1] - VIBRANCE_RANGE[0])),
        "vignette": max(0.0, jitter("vignette", VIGNETTE_RANGE[1] - VIGNETTE_RANGE[0])),
        "detail": max(0.20, jitter("detail", DETAIL_RANGE[1] - DETAIL_RANGE[0])),
    }
    return params


def apply_complex_edit(image, params):
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    arr = rgb.copy()

    gains = np.asarray(params["channel_gains"], dtype=np.float32).reshape(1, 1, 3)
    arr *= gains
    arr *= np.float32(2.0 ** params["exposure"])
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.power(arr, 1.0 / np.float32(params["gamma"]))
    arr = 0.5 + np.float32(params["contrast"]) * (arr - 0.5)

    lum = arr @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    gray = lum[..., None]
    arr = gray + np.float32(params["saturation"]) * (arr - gray)
    sat_strength = np.mean(np.abs(arr - gray), axis=2, keepdims=True)
    arr = arr + np.float32(params["vibrance"]) * (1.0 - sat_strength) * (arr - gray)

    lum = np.clip(arr @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32), 0.0, 1.0)
    shadow_mask = (1.0 - lum) ** 2
    highlight_mask = lum ** 2
    arr = arr + np.float32(params["shadow"]) * shadow_mask[..., None]
    arr = arr - np.float32(params["highlight"]) * highlight_mask[..., None] * (arr - 0.82)

    warmth = np.float32(params["warmth"])
    if warmth != 0:
        warm = np.asarray([1.0, 0.55, -0.55], dtype=np.float32)
        cool = np.asarray([-0.55, 0.0, 1.0], dtype=np.float32)
        arr = arr + warmth * shadow_mask[..., None] * warm * 0.08
        arr = arr + warmth * highlight_mask[..., None] * cool * 0.04

    vignette = np.float32(params["vignette"])
    if vignette > 0:
        h, w = arr.shape[:2]
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        rr = np.sqrt(xx * xx + yy * yy)
        mask = 1.0 - vignette * np.clip((rr - 0.15) / 1.10, 0.0, 1.0) ** 2
        arr *= mask[..., None]

    arr = np.clip(arr, 0.0, 1.0)
    out = Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB")

    detail = params["detail"]
    if abs(detail - 1.0) > 1e-4:
        blurred = out.filter(ImageFilter.GaussianBlur(radius=1.0))
        a = np.asarray(out, dtype=np.float32)
        b = np.asarray(blurred, dtype=np.float32)
        out = Image.fromarray(np.clip(b + detail * (a - b), 0.0, 255.0).round().astype(np.uint8), mode="RGB")
    return out


def load_phase1(path):
    from phase1.model import Model

    if not path.is_file():
        raise FileNotFoundError(f"phase1 权重不存在：{path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("dim", -1)) != 3:
        raise ValueError(f"要求三维 z，当前权重 dim={checkpoint.get('dim')}")
    model = Model(dim=3, width=int(checkpoint["width"]), lut_size=int(checkpoint["lut_size"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE)
    model = model.to(device).eval().requires_grad_(False)
    info = {k: checkpoint.get(k) for k in ("dim", "width", "lut_size", "epoch", "val_psnr", "dataset")}
    return model, device, info


def existing_outputs(root):
    return [p for p in root.iterdir() if re.fullmatch(r"user[1-9]\d*", p.name) or p.name in META_FILES]


def publish(stage, root, old):
    stage, root = stage.resolve(), root.resolve()
    if stage.parent != root:
        raise ValueError("临时结果必须位于所选分组目录内")
    backup, moved_old, moved_new = None, [], []
    try:
        if old:
            backup = root / ("users.previous-" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            backup.mkdir()
            for path in old:
                if path.parent.resolve() != root:
                    raise ValueError("旧结果路径越界")
                path.rename(backup / path.name)
                moved_old.append(path.name)
        for path in sorted(stage.iterdir()):
            destination = root / path.name
            if destination.exists():
                raise FileExistsError(f"目标已存在，停止覆盖：{destination}")
            path.rename(destination)
            moved_new.append(path.name)
    except Exception:
        for name in reversed(moved_new):
            (root / name).rename(stage / name)
        for name in reversed(moved_old):
            (backup / name).rename(root / name)
        if backup and not any(backup.iterdir()):
            backup.rmdir()
        raise
    return backup


def psnr_uint8(pred, target):
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mse = np.mean((pred - target) ** 2) / (255.0 ** 2)
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


@torch.inference_mode()
def encode_and_score(model, device, source_image, target_images):
    source_tensor = to_tensor(source_image).to(device)
    src_batch = torch.stack([source_tensor] * len(target_images), dim=0)
    tgt_batch = torch.stack([to_tensor(t).to(device) for t in target_images], dim=0)
    z = model.encode(src_batch, tgt_batch)
    output = model.decode(src_batch, z)[0].clamp(0.0, 1.0)
    z_np = z.cpu().numpy().astype(np.float32)
    pred_uint8 = (output.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    tgt_uint8 = np.stack([np.asarray(t, dtype=np.uint8) for t in target_images], axis=0)
    psnrs = np.asarray([psnr_uint8(pred_uint8[i], tgt_uint8[i]) for i in range(len(target_images))], dtype=np.float32)
    return z_np, psnrs


def compute_group_stats(group_ids, rows, latents):
    lookup = {gid: i for i, gid in enumerate(group_ids)}
    means = np.empty((NUM_USERS, len(group_ids), 3), dtype=np.float32)
    stds = np.empty_like(means)
    for u in range(NUM_USERS):
        grouped = {gid: [] for gid in group_ids}
        for i, row in enumerate(rows):
            grouped[row["group"]].append(latents[u, i])
        for gid, vecs in grouped.items():
            arr = np.asarray(vecs, dtype=np.float32)
            j = lookup[gid]
            means[u, j] = arr.mean(axis=0)
            stds[u, j] = arr.std(axis=0)
    return means, stds


def generate_one(root, group_ids, rows, grouping_info):
    if NUM_USERS != 1:
        raise ValueError("本版本每套分组固定生成 1 个用户，请将 NUM_USERS 保持为 1")
    checkpoint_path = resolve(CHECKPOINT)
    print(f"自动跟随 group.py：方式 {grouping_info['mode']}（{grouping_info['name']}），目录 {root}", flush=True)
    old = existing_outputs(root)
    if old and not OVERWRITE:
        raise FileExistsError("已有 user* 或用户参数结果；如需重做，请在顶部设置 OVERWRITE=True")

    # random1..3 代表三位不同模拟用户；分类编号相同也不会导致偏好或单图扰动相同。
    dataset_key = f"{grouping_info['mode']}:{grouping_info.get('network_index', root.name)}"
    preferences = sample_preferences(group_ids, dataset_key)
    model, device, checkpoint_info = load_phase1(checkpoint_path)
    users = [f"user{i + 1}" for i in range(NUM_USERS)]
    print(f"{len(rows)} 张原图，{len(group_ids)} 类，{NUM_USERS} 个用户，device={device}", flush=True)

    latents = np.empty((NUM_USERS, len(rows), 3), dtype=np.float32)
    fit_psnr = np.empty((NUM_USERS, len(rows)), dtype=np.float32)

    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".users-", dir=root) as temporary:
        stage = Path(temporary)
        for name in users:
            (stage / name).mkdir()

        image_records = []
        for i, row in enumerate(rows):
            source_digest = sha256(row["path"])
            if row.get('expected_sha256') and source_digest != row['expected_sha256']:
                raise ValueError(f"原图副本 {row['path']} 已改变，请先重新运行 group.py")
            source = rgb_image(row["path"])
            source_rel = row["path"].relative_to(root).as_posix()
            image_records.append({
                "id": row["id"],
                "group": row["group"],
                "source": source_rel,
                "source_sha256": source_digest,
                "size": [source.width, source.height],
            })

            target_images = []
            for u, user_name in enumerate(users):
                params = perturb_preferences(preferences[u][row["group"]], row["id"], u, row["group"], dataset_key)
                target = apply_complex_edit(source, params)
                target.save(stage / user_name / f"{row['id']}.png")
                target_images.append(target)

            z_np, psnr_np = encode_and_score(model, device, source, target_images)
            for u in range(NUM_USERS):
                latents[u, i] = z_np[u]
                fit_psnr[u, i] = psnr_np[u]

            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                summary = ", ".join(f"{users[u]} {fit_psnr[u, i]:.2f}dB" for u in range(NUM_USERS))
                print(f"完成 {i + 1}/{len(rows)} 张原图；当前 z 拟合 PSNR：{summary}", flush=True)

        means, stds = compute_group_stats(group_ids, rows, latents)

        np.savez_compressed(stage / META_FILES[1],
                            users=np.asarray(users),
                            ids=np.asarray([r["id"] for r in rows]),
                            groups=np.asarray(group_ids),
                            image_groups=np.asarray([r["group"] for r in rows]),
                            zmean=means,
                            zstd=stds,
                            z=latents,
                            fit_psnr=fit_psnr)

        summary_users = []
        group_lookup = {gid: idx for idx, gid in enumerate(group_ids)}
        for u, user_name in enumerate(users):
            group_summaries = {}
            for gid in group_ids:
                idxs = [k for k, r in enumerate(rows) if r["group"] == gid]
                group_summaries[gid] = {
                    "preference": preferences[u][gid],
                    "fitted_z_mean": means[u, group_lookup[gid]].tolist(),
                    "fitted_z_std": stds[u, group_lookup[gid]].tolist(),
                    "fit_psnr_mean": float(fit_psnr[u, idxs].mean()),
                    "fit_psnr_min": float(fit_psnr[u, idxs].min()),
                    "fit_psnr_max": float(fit_psnr[u, idxs].max()),
                }
            summary_users.append({
                "id": user_name,
                "preferences": group_summaries,
                "fit_psnr_mean": float(fit_psnr[u].mean()),
                "fit_psnr_min": float(fit_psnr[u].min()),
                "fit_psnr_max": float(fit_psnr[u].max()),
            })

        metadata = {
            "seed": SEED,
            "group_dir": str(root),
            "group_mode": grouping_info["mode"],
            "grouping": grouping_info,
            "num_users": NUM_USERS,
            "images_per_user": len(rows),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_info": checkpoint_info,
            "z_dimension": 3,
            "generation_method": "Complex handcrafted photo edits per user/category, then one-pass phase1 encode",
            "latent_file": META_FILES[1],
            "latent_axes": {
                "z": ["user", "image", "dimension"],
                "zmean_zstd": ["user", "group", "dimension"],
                "fit_psnr": ["user", "image"],
            },
            "edit_parameter_ranges": {
                "exposure": EXPOSURE_RANGE,
                "contrast": CONTRAST_RANGE,
                "saturation": SATURATION_RANGE,
                "gamma": GAMMA_RANGE,
                "shadow": SHADOW_RANGE,
                "highlight": HIGHLIGHT_RANGE,
                "channel_gains": CHANNEL_GAIN_RANGE,
                "warmth": WARMTH_RANGE,
                "vibrance": VIBRANCE_RANGE,
                "vignette": VIGNETTE_RANGE,
                "detail": DETAIL_RANGE,
                "noise_scale": NOISE_SCALE,
            },
            "users": summary_users,
            "images": image_records,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (stage / META_FILES[0]).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        for name in users:
            if len(list((stage / name).glob("*.png"))) != len(rows):
                raise RuntimeError(f"{name} 图片数不完整，未提交生成结果")
        if sha256(Path(grouping_info['manifest'])) != grouping_info['manifest_sha256']:
            raise RuntimeError('生成期间分组清单改变，未发布用户结果，请重新运行 user.py')
        backup = publish(stage, root, old)

    print(f"完成：{root} 下 {', '.join(users)}，每个用户 {len(rows)} 张图片。")
    print(f"已保存 z 与 fit_psnr 到 {root / META_FILES[1]}")
    if backup:
        print(f"旧用户结果备份：{backup}")
    for u, name in enumerate(users):
        print(f"{name} 平均 z 拟合 PSNR: {fit_psnr[u].mean():.4f} dB")


def generate():
    datasets = collect_current_groups()
    print(f"自动跟随 group.py：共 {len(datasets)} 套分组；每套生成 1 个用户。", flush=True)
    for root, group_ids, rows, grouping_info in datasets:
        print(f"开始：方式 {grouping_info['mode']}（{grouping_info['name']}），目录 {root}", flush=True)
        generate_one(root, group_ids, rows, grouping_info)


def ensure_expert_groups(raw_dir=None, group_dir=None, experts="ABCDE", force=False):
    """每个专家单独对原始三维 z 做 KMeans，输出该专家的 1..6 原图文件夹。"""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    raw = resolve(raw_dir or RAW_DIR)
    output = resolve(group_dir or EXPERT_GROUP_DIR)
    if output == raw or raw in output.parents or output in raw.parents:
        raise ValueError("专家分组输出与原始数据目录不能互相包含")
    if EXPERT_GROUPS < 2: raise ValueError("EXPERT_GROUPS 至少为 2")
    metadata = json.loads((raw / "user_parameters.json").read_text(encoding="utf-8"))
    cache_hash = sha256(raw / "user_latents.npz")
    if metadata.get("latent_sha256") != cache_hash:
        raise ValueError("专家 z 缓存与元数据不一致，请重新生成 z")
    with np.load(raw / "user_latents.npz", allow_pickle=False) as data:
        users, ids = data["users"].astype(str).tolist(), data["ids"].astype(str).tolist()
        z = data["z"].copy()
        valid = data["valid"].copy() if "valid" in data else np.isfinite(z).all(2)
    if ids != [str(row["id"]) for row in metadata["images"]]: raise ValueError("专家 z 图片编号不一致")
    if z.shape != (len(users), len(ids), 3) or valid.shape != z.shape[:2]: raise ValueError("专家 z 形状不正确")
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for expert in experts:
        if expert not in "ABCDE" or expert not in users: raise ValueError(f"未知专家：{expert}")
        u = users.index(expert)
        indices = [i for i, row in enumerate(metadata["images"])
                   if valid[u, i] and np.isfinite(z[u, i]).all() and (raw / row["source"]).is_file()
                   and expert in row.get("targets", {}) and (raw / row["targets"][expert]).is_file()]
        values = z[u, indices].astype(np.float64)
        if len(values) < EXPERT_GROUPS or len(np.unique(values, axis=0)) < EXPERT_GROUPS:
            raise ValueError(f"专家 {expert} 的不同有效 z 不足 {EXPERT_GROUPS} 个")
        config = dict(method="per-expert KMeans on raw 3D z", n_groups=EXPERT_GROUPS, seed=SEED, n_init=20,
                      latent_sha256=cache_hash, checkpoint_sha256=metadata["checkpoint_sha256"])
        destination = output / expert
        manifest_path = destination / "group_expert.json"
        if destination.is_symlink(): raise ValueError(f"分组目录不能为符号链接：{destination}")
        if manifest_path.is_file() and not force:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (previous.get("expert") == expert and previous.get("config") == config
                    and [r["id"] for r in previous["images"]] == [ids[i] for i in indices]
                    and all((destination / r["file"]).is_file() for r in previous["images"])):
                print(f"复用专家 {expert} 的 {EXPERT_GROUPS} 个 z 分组", flush=True)
                results[expert] = manifest_path
                continue
        if destination.exists() and (not destination.is_dir() or not manifest_path.is_file()):
            raise ValueError(f"输出已存在且不是本程序的专家分组：{destination}")
        with threadpool_limits(limits=1):
            clustering = KMeans(n_clusters=EXPERT_GROUPS, n_init=20, random_state=SEED).fit(values)
        centers = clustering.cluster_centers_
        # 以中心的 z1/z2/z3 顺序固定类别编号；不同专家的编号不代表相同语义。
        order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
        mapping = np.empty(EXPERT_GROUPS, np.int64); mapping[order] = np.arange(1, EXPERT_GROUPS + 1)
        labels = mapping[clustering.labels_]
        rows = [dict(id=ids[i], group=str(int(label)), source=str(raw / metadata["images"][i]["source"]),
                     target=str(raw / metadata["images"][i]["targets"][expert]), file=f"{label}/{ids[i]}.png")
                for i, label in zip(indices, labels)]
        categories = []
        for group in range(1, EXPERT_GROUPS + 1):
            subset = values[labels == group]
            categories.append(dict(id=str(group), count=len(subset), zmean=subset.mean(0).tolist(), zstd=subset.std(0).tolist()))
        manifest = dict(expert=expert, config=config, total=len(rows), categories=categories, images=rows,
                        created_at_utc=datetime.now(timezone.utc).isoformat(),
                        note="使用该专家全部有效 z 独立聚类；类别表示编辑空间簇，不是语义类别。训练基线的组均值另外仅用训练集计算。")
        # 旧分组只临时移入本次 staging；提交成功后自动清理，失败则恢复。
        with tempfile.TemporaryDirectory(prefix=f".expert-{expert}-", dir=output) as temporary:
            stage = Path(temporary) / "new"; stage.mkdir()
            for group in range(1, EXPERT_GROUPS + 1): (stage / str(group)).mkdir()
            for row in rows: shutil.copy2(row["source"], stage / row["file"])
            (stage / "group_expert.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if sha256(raw / "user_latents.npz") != cache_hash: raise RuntimeError("聚类期间 z 缓存改变，请重新运行")
            backup = Path(temporary) / "old"
            if destination.exists(): destination.rename(backup)
            try: stage.rename(destination)
            except OSError:
                if backup.exists() and not destination.exists(): backup.rename(destination)
                raise
        results[expert] = manifest_path
        print(f"专家 {expert}：{len(rows)} 张，分组数量 {[c['count'] for c in categories]}；{destination}", flush=True)
    return results


def generate_expert_groups():
    # 使用 train.py 的同一套专家 z 缓存生成/校验逻辑，避免维护第二套 z。
    import train as training
    original_raw, original_checkpoint = training.RAW_DIR, training.PHASE1_CHECKPOINT
    training.RAW_DIR, training.PHASE1_CHECKPOINT = str(resolve(RAW_DIR)), str(resolve(CHECKPOINT))
    try:
        model, device, _ = load_phase1(resolve(CHECKPOINT))
        training.ensure_expert_cache(model, device)
    finally:
        training.RAW_DIR, training.PHASE1_CHECKPOINT = original_raw, original_checkpoint
    ensure_expert_groups()


def main():
    parser = argparse.ArgumentParser(description='自动跟随 group.py 最近成功输出，生成三个模拟用户')
    parser.add_argument('--experts', action='store_true', help='单独给 A-E 各自的 z 聚六组，不生成模拟用户')
    args = parser.parse_args()
    if args.experts: generate_expert_groups()
    else: generate()


if __name__ == "__main__":
    main()
