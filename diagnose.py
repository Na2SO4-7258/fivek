# ========================= 超参数 =========================
GROUP_DIR = "data/semantic"          # 可改成 data/raw 分析真实专家
CHECKPOINT = "phase1/out/model/best.pt"
USERS = None                         # None=全部；例如 ["user1", "user2"]
REFRESH_ALL_CACHES = True             # 权重/图片改变时，在原位置更新全部现用 z 缓存
DEVICE = "auto"
RENDER_BATCH = 4
PERMUTATIONS = 999
SEED = 42
COMPUTE_SSIM = True
SAVE_PLOTS = True
DISTANCE_THRESHOLDS = (0.25, 0.5, 1.0)
OUTPUT_DIR = None                    # None=GROUP_DIR/diagnosis
# ==========================================================
"""运行 python diagnose.py；复用本项目 analyze_z.py、user.py、phase1。
依赖 torch、numpy、Pillow、scipy、scikit-learn、scikit-image、matplotlib。
类别平均 z 是当前用户该类别所有有效图片 z 的均值；这是数据描述，不是测试集预测。
额外计算留一类别均值、用户全局均值，均不改变原图或修图。所有距离为原始 z 欧氏距离。
输出 report.txt/json、users.csv、groups.csv、images.csv、centers.csv 和可选图表。
"""
import csv, hashlib, io, json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import pdist
from skimage.metrics import structural_similarity

import analyze_z as az
from user import rgb_image
from phase1.model import Model

ROOT = Path(__file__).resolve().parent
METHODS = ('original', 'fitted_z', 'class_mean_z', 'loo_class_mean_z', 'user_mean_z', 'class_mean_vs_fitted')
LABELS = dict(original='原图 vs 修图', fitted_z='逐图 z 重建 vs 修图',
              class_mean_z='类别平均 z 重建 vs 修图', loo_class_mean_z='留一类别平均 z 重建 vs 修图',
              user_mean_z='用户全局平均 z 重建 vs 修图', class_mean_vs_fitted='类别均值重建 vs 逐图 z 重建')


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def stats(values, weights=None):
    """权重分位数对应经验分布；无权重使用 numpy 线性分位数。保留完美匹配的 +inf。"""
    a = np.asarray(values, np.float64)
    mask = ~np.isnan(a)
    a = a[mask]
    if not len(a): return dict(n=0, mean=None, std=None, min=None, p05=None, p50=None, p90=None, p95=None, max=None)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, np.float64)[mask]
    order = np.argsort(a); sorted_a, sorted_w = a[order], w[order]
    if weights is None and np.isfinite(a).all():
        quantiles = np.quantile(a, [.05, .5, .9, .95])
    else:
        cumulative = np.cumsum(sorted_w) / sorted_w.sum()
        quantiles = [sorted_a[min(np.searchsorted(cumulative, q), len(a) - 1)] for q in [.05, .5, .9, .95]]
    return dict(n=len(a), mean=float(np.average(a, weights=w)),
                std=float(np.sqrt(np.average((a - np.average(a, weights=w)) ** 2, weights=w))) if np.isfinite(a).all() else None,
                min=float(a.min()), **dict(zip(('p05', 'p50', 'p90', 'p95'), map(float, quantiles))), max=float(a.max()))


def add_stats(row, prefix, values, weights=None):
    row.update({f'{prefix}_{k}': v for k, v in stats(values, weights).items()})


def image_metrics(prediction, target):
    diff = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.square(diff).mean())
    result = dict(psnr_db=10 * math.log10(255. ** 2 / mse) if mse else float('inf'),
                  mae_255=float(np.abs(diff).mean()), rmse_255=math.sqrt(mse), mse_255=mse)
    result['ssim'] = None
    if COMPUTE_SSIM:
        win = min(11, *target.shape[:2]); win -= 1 - win % 2
        if win >= 3:
            result['ssim'] = float(structural_similarity(prediction, target, channel_axis=-1, data_range=255,
                                   gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=win))
    return result


def mean_excluding(z, count, mean):
    return (mean * count - z) / (count - 1) if count > 1 else None


def latent_stats(z, labels, seed):
    base, group_rows, _ = az.analyze_user(z, labels, seed)
    z = z.astype(np.float64)
    radii, all_distances, all_weights = [], [], []
    centers = {g: z[labels == g].mean(0) for g in np.unique(labels)}
    residual = z - np.stack([centers[g] for g in labels])
    for row in group_rows:
        g = row['group']; v = z[labels == g]; n = len(v)
        radius = np.linalg.norm(v - centers[g], axis=1)
        distances = pdist(v)
        radii.extend(radius)
        all_distances.extend(distances)
        all_weights.extend(np.full(len(distances), 1 / (n - 1)) if n > 1 else [])
        add_stats(row, 'within_pair', distances)
        add_stats(row, 'center_radius', radius)
        others = [np.linalg.norm(centers[g] - center) for key, center in centers.items() if key != g]
        row['nearest_other_center_distance'] = float(min(others)) if others else None
        row['residual_per_dim_rms'] = float(np.sqrt(np.square(v - centers[g]).mean()))
        for j in range(3):
            row.update({f'z{j+1}_{k}': val for k, val in stats(v[:, j]).items() if k != 'n'})
        for i in range(3):
            for j in range(i, 3):
                row[f'cov_z{i+1}_z{j+1}'] = float(np.mean((v[:, i] - centers[g][i]) * (v[:, j] - centers[g][j])))
        for threshold in DISTANCE_THRESHOLDS:
            row[f'pair_fraction_le_{threshold:g}'] = float(np.mean(distances <= threshold)) if len(distances) else None
    add_stats(base, 'within_pair', all_distances, all_weights)
    add_stats(base, 'center_radius', radii)
    base['residual_per_dim_rms'] = float(np.sqrt(np.square(residual).mean()))
    base['within_variance_fraction'] = 1 - base['r_squared'] if base['r_squared'] is not None else None
    for j in range(3):
        base[f'residual_z{j+1}_rms'] = float(np.sqrt(np.square(residual[:, j]).mean()))
        base.update({f'z{j+1}_{k}': val for k, val in stats(z[:, j]).items() if k != 'n'})
    ds, weights = np.asarray(all_distances), np.asarray(all_weights)
    for threshold in DISTANCE_THRESHOLDS:
        base[f'pair_fraction_le_{threshold:g}'] = float(weights[ds <= threshold].sum() / weights.sum()) if len(ds) else None
    center_rows = []
    groups = sorted(centers)
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            center_rows.append(dict(group_a=left, group_b=right, center_distance=float(np.linalg.norm(centers[left] - centers[right]))))
    return base, group_rows, center_rows


@torch.inference_mode()
def render_metrics(model, device, root, a, m, users, valid):
    z = a['z'].astype(np.float64); labels = a['image_groups'].astype(str)
    rows, signature = az.pairs_and_signature(root, a, m)
    group_means, counts, user_means = {}, {}, {}
    for u in users:
        if not valid[u].any(): continue
        user_means[u] = z[u, valid[u]].mean(0)
        for group in np.unique(labels[valid[u]]):
            mask = valid[u] & (labels == group)
            group_means[u, group], counts[u, group] = z[u, mask].mean(0), int(mask.sum())
    records = []
    for i, (source, target_paths) in enumerate(rows):
        active = [u for u in users if valid[u, i]]
        if active:
            original = rgb_image(source)
            pixels = np.asarray(original, np.uint8)
            targets, jobs, vectors = {}, [], []
            for u in active:
                target = rgb_image(target_paths[u])
                if target.size != original.size: target = target.resize(original.size, Image.Resampling.LANCZOS)
                targets[u] = np.asarray(target, np.uint8)
                mean = group_means[u, labels[i]]
                variants = dict(fitted_z=z[u, i], class_mean_z=mean,
                                loo_class_mean_z=mean_excluding(z[u, i], counts[u, labels[i]], mean), user_mean_z=user_means[u])
                for key, vector in variants.items():
                    if vector is not None: jobs.append((u, key)); vectors.append(vector)
            source_tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.
            rendered = {}
            for start in range(0, len(jobs), RENDER_BATCH):
                batch = torch.as_tensor(np.asarray(vectors[start:start + RENDER_BATCH]), dtype=torch.float32, device=device)
                out = model.decode(source_tensor.expand(len(batch), -1, -1, -1).contiguous(), batch)[0]
                if not torch.isfinite(out).all(): raise ValueError(f'解码无效：{source}')
                output = (out.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                rendered.update(zip(jobs[start:start + RENDER_BATCH], output))
            for u in active:
                r = dict(user=str(a['users'][u]), id=str(a['ids'][i]), group=labels[i], source=str(source), target=str(target_paths[u]))
                for j in range(3): r[f'z{j+1}'] = float(z[u, i, j])
                r['center_radius'] = float(np.linalg.norm(z[u, i] - group_means[u, labels[i]]))
                for method in METHODS:
                    pred = pixels if method == 'original' else rendered.get((u, 'class_mean_z' if method == 'class_mean_vs_fitted' else method))
                    target = rendered[u, 'fitted_z'] if method == 'class_mean_vs_fitted' else targets[u]
                    values = image_metrics(pred, target) if pred is not None else {k: None for k in ('psnr_db','ssim','mae_255','rmse_255','mse_255')}
                    r.update({f'{method}_{key}': value for key, value in values.items()})
                # 直接比较 MSE，避免把同为完美匹配的 inf-inf 当作下降。
                fitted, mean = r['fitted_z_mse_255'], r['class_mean_z_mse_255']
                r['mean_psnr_loss_db'] = 10 * math.log10(mean / fitted) if fitted > 0 and mean > 0 else (0. if mean == fitted else (float('inf') if fitted == 0 else -float('inf')))
                r['mean_minus_fitted_mae_255'] = r['class_mean_z_mae_255'] - r['fitted_z_mae_255']
                r['mean_better_than_original'] = r['class_mean_z_mse_255'] < r['original_mse_255']
                r['mean_better_than_fitted'] = mean < fitted
                records.append(r)
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f'图像诊断 {i+1}/{len(rows)}，完成 {len(records)} 个有效图片对', flush=True)
    if az.pairs_and_signature(root, a, m)[1] != signature: raise RuntimeError('诊断期间图片改变，请重新运行')
    return records


def attach_image_stats(row, records):
    for method in METHODS:
        for metric in ('psnr_db', 'ssim', 'mae_255', 'rmse_255'):
            add_stats(row, f'{method}_{metric}', [r[f'{method}_{metric}'] for r in records if r[f'{method}_{metric}'] is not None])
        mses = [r[f'{method}_mse_255'] for r in records if r[f'{method}_mse_255'] is not None]
        average_mse = float(np.mean(mses)) if mses else None
        row[f'{method}_psnr_from_mean_mse_db'] = (10 * math.log10(255.**2 / average_mse) if average_mse else float('inf')) if average_mse is not None else None
    add_stats(row, 'mean_psnr_loss_db', [r['mean_psnr_loss_db'] for r in records])
    add_stats(row, 'mean_minus_fitted_mae_255', [r['mean_minus_fitted_mae_255'] for r in records])
    for margin in (0.5, 1., 2., 3.):
        row[f'mean_psnr_loss_le_{margin:g}db_fraction'] = float(np.mean([r['mean_psnr_loss_db'] <= margin for r in records]))
    row['mean_better_than_original_fraction'] = float(np.mean([r['mean_better_than_original'] for r in records]))
    row['mean_better_than_fitted_fraction'] = float(np.mean([r['mean_better_than_fitted'] for r in records]))


def clean_json(value):
    if isinstance(value, dict): return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, np.generic): return value.item()
    return value


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig'); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def fmt(value, digits=3):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def report_text(report):
    lines = ['数据集 z 与图像重建诊断', f"数据：{report['group_dir']}",
             f"权重：epoch={report['checkpoint_info']['epoch']}，sha256={report['checkpoint_sha256']}",
             '单位：z 为未标准化的三维欧氏距离；PSNR 为 dB（越高越好）；SSIM 越高越好；MAE/RMSE 为 0–255 RGB 灰度级（越低越好）。',
             '类别平均 z：同用户、同类别的有效图片均值（包含自身）；留一类别均值：排除当前图片。',
             '均值由当前被诊断图片计算，因此报告是样本内描述，不是未见图片的泛化成绩。',
             '总体类内配对距离：先按图片等权，再在其同类其他图片中等权；单样本类不参与配对距离。',
             '半径为各图片距所属类别中心的距离。R² 为类别中心解释的总平方离差比例，不是相似度百分比。',
             '图像 PSNR 默认逐图算 dB 后等权平均；CSV 另列先平均逐图 MSE 再换算的 PSNR。',
             '所有重建均 clamp 到 [0,1] 后量化为 uint8，与 train.py 的图像评估一致；修图尺寸不同时对齐原图。',
             'SSIM 使用 RGB 通道均值、Gaussian sigma=1.5、总体协方差；正常图片窗口为 11。',
             '逐图 z 来自 O 和目标修图共同编码，是重建参照，不是仅凭原图预测的成绩。',
             'R² 置换检验固定类别人数，Holm p 校正本次被诊断用户；不把类别结构直接视作偏好独立于内容的证明。', '']
    for r in report['users']:
        lines += [f"{r['user']}：有效图片 {r['n']}，类别 {r['groups']}，跳过 {r['skipped']}",
                  f"  同类配对距离：均值 {fmt(r['within_pair_mean'])}，中位数 {fmt(r['within_pair_p50'])}，P90 {fmt(r['within_pair_p90'])}，P95 {fmt(r['within_pair_p95'])}；异类均值 {fmt(r['between_distance'])}；比值 {fmt(r['distance_ratio'])}",
                  f"  类别中心半径：均值 {fmt(r['center_radius_mean'])}，P90 {fmt(r['center_radius_p90'])}，P95 {fmt(r['center_radius_p95'])}；每维残差 RMS {fmt(r['residual_per_dim_rms'])}",
                  f"  类别解释 R²={fmt(r['r_squared'])}，类内剩余比例={fmt(r['within_variance_fraction'])}，silhouette={fmt(r['silhouette'])}，Holm p={fmt(r['holm_p'],4)}"]
        for method in METHODS:
            lines.append(f"  {LABELS[method]}：PSNR {fmt(r[method+'_psnr_db_mean'])} ± {fmt(r[method+'_psnr_db_std'])}，P05/P50/P95={fmt(r[method+'_psnr_db_p05'])}/{fmt(r[method+'_psnr_db_p50'])}/{fmt(r[method+'_psnr_db_p95'])}；SSIM {fmt(r[method+'_ssim_mean'],4)}；MAE {fmt(r[method+'_mae_255_mean'])}；RMSE {fmt(r[method+'_rmse_255_mean'])}")
        lines += [f"  换成类别均值 z 后的 PSNR 损失：均值 {fmt(r['mean_psnr_loss_db_mean'])} dB，P90 {fmt(r['mean_psnr_loss_db_p90'])} dB；损失≤1 dB 比例 {fmt(100*r['mean_psnr_loss_le_1db_fraction'],1)}%；≤3 dB 比例 {fmt(100*r['mean_psnr_loss_le_3db_fraction'],1)}%（负损失表示改善）",
                  f"  类别均值修图优于原图的比例：{fmt(100*r['mean_better_than_original_fraction'],1)}%", '  各类别：']
        for g in [g for g in report['groups'] if g['user'] == r['user']]:
            means = [g[f'z{j}_mean'] for j in range(1,4)]; stds = [g[f'z{j}_std'] for j in range(1,4)]
            lines += [f"    {g['name']} (n={g['n']}) zmean={np.round(means,4).tolist()}，zstd={np.round(stds,4).tolist()}",
                      f"      类内均值/P90={fmt(g['within_pair_mean'])}/{fmt(g['within_pair_p90'])}，中心半径P90={fmt(g['center_radius_p90'])}，silhouette={fmt(g['silhouette'])}",
                      f"      PSNR 原图/逐图z/类别均值/留一均值={fmt(g['original_psnr_db_mean'])}/{fmt(g['fitted_z_psnr_db_mean'])}/{fmt(g['class_mean_z_psnr_db_mean'])}/{fmt(g['loo_class_mean_z_psnr_db_mean'])}；均值替换损失={fmt(g['mean_psnr_loss_db_mean'])} dB"]
        lines.append('')
    if report['checkpoint_changed_during_run']: lines.append('运行期间磁盘权重发生改变；本报告使用上面记录的同一个权重快照。')
    lines += ['users.csv：每用户完整统计；groups.csv：每用户×类别统计；images.csv：逐图指标，便于定位异常；centers.csv：各类别中心间距离。',
              'PSNR/SSIM 只描述像素与结构接近程度，不代表审美质量或严格的感知等价；Inf 表示像素完全一致。']
    return '\n'.join(lines) + '\n'


def plot_summary(out, users):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(users)); width=.24
    for offset, method in enumerate(METHODS[:3]):
        axes[0].bar(x+(offset-1)*width, [r[method+'_psnr_db_mean'] for r in users], width, label=LABELS[method])
    axes[0].set(xticks=x, xticklabels=[r['user'] for r in users], ylabel='平均 PSNR (dB)', title='图像重建效果')
    axes[0].legend(fontsize=8)
    for offset, key in enumerate(('within_pair_mean','within_pair_p90','center_radius_p90')):
        axes[1].bar(x+(offset-1)*width, [r[key] if r[key] is not None else np.nan for r in users], width,
                    label=('同类距离均值','同类距离 P90','类别中心半径 P90')[offset])
    axes[1].set(xticks=x, xticklabels=[r['user'] for r in users], ylabel='原始 z 欧氏距离', title='类内集中程度')
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'summary.png', dpi=160); plt.close(fig)


def main():
    if RENDER_BATCH < 1 or PERMUTATIONS < 1: raise ValueError('RENDER_BATCH / PERMUTATIONS 至少为 1')
    root = resolve(GROUP_DIR); out = resolve(OUTPUT_DIR) if OUTPUT_DIR else root/'diagnosis'
    content = resolve(CHECKPOINT).read_bytes(); checkpoint_hash = hashlib.sha256(content).hexdigest()
    ckpt = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True); del content
    if int(ckpt['dim']) != 3: raise ValueError('要求三维 phase1')
    info = {k: ckpt.get(k) for k in ('dim','width','lut_size','epoch','val_psnr','dataset')}
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if DEVICE == 'auto' else DEVICE)
    model = Model(3, int(ckpt['width']), int(ckpt['lut_size'])).to(device).eval().requires_grad_(False)
    model.load_state_dict(ckpt['model'], strict=True); del ckpt
    encoder_hash = hashlib.sha256(''.join(az.sha(ROOT/'phase1'/p) for p in ('model.py','config.py')).encode()).hexdigest()
    az.CHECKPOINT = str(resolve(CHECKPOINT)); az.STANDARDIZE = False; az.PERMUTATIONS = PERMUTATIONS
    az.FORCE_REENCODE = False
    caches = {root}
    if REFRESH_ALL_CACHES:
        for p in (ROOT/'data').rglob('user_latents.npz'):
            if any(s.startswith('.') or '.previous-' in s or s in ('z_analysis','diagnosis') for s in p.relative_to(ROOT/'data').parts): continue
            caches.add(p.parent)
    print(f'诊断权重 epoch={info["epoch"]}，device={device}，sha256={checkpoint_hash}', flush=True)
    for p in sorted(caches): az.refresh_cache(p, model, device, checkpoint_hash, info, encoder_hash)
    a, m = az.load_cache(root); cache_hash = az.sha(root/'user_latents.npz')
    names = a['users'].astype(str).tolist()
    requested = names if USERS is None else ([USERS] if isinstance(USERS, str) else list(USERS))
    if not requested or not set(requested).issubset(names): raise ValueError(f'用户必须来自 {names}')
    indices = [names.index(u) for u in dict.fromkeys(requested)]
    valid = np.isfinite(a['z']).all(2)
    if 'valid' in a: valid &= a['valid']
    pairs, _ = az.pairs_and_signature(root, a, m)
    for i, (source, targets) in enumerate(pairs):
        for u in indices: valid[u, i] &= source.is_file() and targets[u].is_file()
    labels = a['image_groups'].astype(str)
    category_names = {}
    for p in root.glob('group_*.json'):
        for c in json.loads(p.read_text(encoding='utf-8')).get('categories', []): category_names[str(c['id'])] = c['name']
    users, groups, centers = [], [], []
    for u in indices:
        if not valid[u].any(): print(f'{names[u]} 没有有效图片对，跳过'); continue
        summary, details, center_rows = latent_stats(a['z'][u, valid[u]], labels[valid[u]], SEED + u)
        users.append(dict(user=names[u], **summary, skipped=int((~valid[u]).sum())))
        groups.extend(dict(user=names[u], name=category_names.get(r['group'], r['group']), **r) for r in details)
        centers.extend(dict(user=names[u], **r) for r in center_rows)
    if not users: raise ValueError('没有有效用户图片对')
    az.holm(users)
    images = render_metrics(model, device, root, a, m, indices, valid)
    if az.sha(root/'user_latents.npz') != cache_hash: raise RuntimeError('诊断期间 z 缓存改变，请重新运行')
    for row in users: attach_image_stats(row, [r for r in images if r['user'] == row['user']])
    for row in groups: attach_image_stats(row, [r for r in images if r['user'] == row['user'] and r['group'] == row['group']])
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), group_dir=str(root), checkpoint_info=info,
                  checkpoint_sha256=checkpoint_hash, latent_sha256=cache_hash,
                  checkpoint_changed_during_run=az.sha(resolve(CHECKPOINT)) != checkpoint_hash,
                  permutations=PERMUTATIONS, seed=SEED, ssim=COMPUTE_SSIM,
                  mean_definition='per-user per-class empirical mean, including self; also report leave-one-out mean',
                  users=users, groups=groups, centers=centers)
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('users.csv',users),('groups.csv',groups),('images.csv',images),('centers.csv',centers)]: write_csv(out/filename,rows)
    (out/'report.json').write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    text = report_text(report); (out/'report.txt').write_text(text, encoding='utf-8')
    if SAVE_PLOTS: plot_summary(out, users)
    print(text, flush=True); print(f'诊断输出：{out}', flush=True)


if __name__ == '__main__': main()
