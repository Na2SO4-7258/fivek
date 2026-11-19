# ======================== 超参数 ========================
GROUP_DIR = "data/semantic"       # 分析哪个分类下的用户
CHECKPOINT = "phase1/out/model/best.pt"
REFRESH_ALL_CACHES = True          # 更新 data 下所有当前 z 缓存，不进入历史备份
FORCE_REENCODE = False             # True=即使权重、图片未变也重新编码
STANDARDIZE = False                # False=原始 z 欧氏距离；True=每用户各维标准化
PERMUTATIONS = 999                 # 保持类别人数不变，随机打乱标签
SEED = 42
DEVICE = "auto"
# ========================================================
"""python analyze_z.py；依赖 torch、numpy、Pillow、scipy、scikit-learn、matplotlib。
直接更新原位置 user_latents.npz / user_parameters.json；不修改图片或分组特征。
分析结果存 GROUP_DIR/z_analysis。类内/类间距离比 < 1 表示同类更近；
silhouette 越接近 1 越分离，约 0 表示重叠，负值表示更靠近其他类。
R²=类别中心之间的平方离差/总平方离差；置换检验检查标签与 z 的关联。
这些指标衡量 z 的类别结构，不能单独证明用户偏好不受原图内容影响。
"""
import csv, hashlib, io, json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples

ROOT = Path(__file__).resolve().parent


def resolve(path):
    p = Path(path)
    return (p if p.is_absolute() else ROOT / p).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def load_cache(root):
    m = json.loads((root / 'user_parameters.json').read_text(encoding='utf-8'))
    with np.load(root / 'user_latents.npz', allow_pickle=False) as d:
        a = {k: d[k].copy() for k in d.files}
    users, ids = a['users'].astype(str).tolist(), a['ids'].astype(str).tolist()
    if len(set(users)) != len(users) or len(set(ids)) != len(ids):
        raise ValueError(f'重复用户或图片编号：{root}')
    if a['z'].shape != (len(users), len(ids), 3): raise ValueError(f'z 形状错误：{root}')
    if ids != [str(r['id']) for r in m['images']]: raise ValueError(f'缓存与元数据编号不一致：{root}')
    if a['image_groups'].astype(str).tolist() != [str(r['group']) for r in m['images']]:
        raise ValueError(f'缓存与元数据类别不一致：{root}')
    if not set(a['image_groups'].astype(str)).issubset(a['groups'].astype(str)):
        raise ValueError(f'未知类别：{root}')
    return a, m


def pairs_and_signature(root, a, m):
    rows, stats = [], {}
    for r in m['images']:
        source = root / r['source']
        targets = [root / r['targets'][u] if 'targets' in r and u in r['targets']
                   else root / u / f"{r['id']}.png" for u in a['users'].astype(str)]
        rows.append((source, targets))
        for p in [source, *targets]:
            if str(p) not in stats:
                st = p.stat() if p.is_file() else None
                stats[str(p)] = (st.st_size, st.st_mtime_ns) if st else None
    return rows, hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()


def refresh_cache(root, model, device, checkpoint_hash, checkpoint_info, encoder_hash):
    from PIL import Image
    from user import rgb_image, encode_and_score
    a, m = load_cache(root)
    rows, signature = pairs_and_signature(root, a, m)
    npz_path = root / 'user_latents.npz'
    if (not FORCE_REENCODE and m.get('checkpoint_sha256') == checkpoint_hash
            and m.get('z_input_signature') == signature and m.get('encoder_signature') == encoder_hash
            and m.get('latent_sha256') == sha(npz_path)):
        print(f'复用已更新的 z：{root}', flush=True)
        return
    users, groups = a['users'].astype(str).tolist(), a['groups'].astype(str).tolist()
    z = np.full(a['z'].shape, np.nan, np.float32)
    fit = np.full(z.shape[:2], np.nan, np.float32)
    valid = np.zeros(z.shape[:2], bool)
    for i, (source, targets) in enumerate(rows):
        if source.is_file():
            original = rgb_image(source)
            indices = [u for u, p in enumerate(targets) if p.is_file()]
            images = []
            for u in indices:
                target = rgb_image(targets[u])
                if target.size != original.size: target = target.resize(original.size, Image.Resampling.LANCZOS)
                images.append(target)
            if images:
                zi, pi = encode_and_score(model, device, original, images)
                if not np.isfinite(zi).all(): raise ValueError(f'无效编码：{source}')
                z[indices, i], fit[indices, i], valid[indices, i] = zi, pi, True
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f'{root.name} 编码 {i + 1}/{len(rows)}，有效图片对 {valid.sum()}', flush=True)
    if not valid.any(): raise ValueError(f'没有可编码的图片对：{root}')
    if pairs_and_signature(root, a, m)[1] != signature: raise RuntimeError('编码期间输入图片改变，请重新运行')
    means = np.full((len(users), len(groups), 3), np.nan, np.float32)
    stds = np.full_like(means, np.nan)
    for u, name in enumerate(users):
        entry = next((v for v in m['users'] if v['id'] == name), None)
        for g, group in enumerate(groups):
            mask = valid[u] & (a['image_groups'].astype(str) == group)
            if not mask.any(): continue
            means[u, g], stds[u, g] = z[u, mask].mean(0), z[u, mask].std(0)
            preference = entry.get('preferences', {}).get(group) if entry else None
            if preference is not None:
                preference.update(fitted_z_mean=means[u, g].tolist(), fitted_z_std=stds[u, g].tolist(),
                                  fit_psnr_mean=float(fit[u, mask].mean()), fit_psnr_min=float(fit[u, mask].min()),
                                  fit_psnr_max=float(fit[u, mask].max()))
        if entry is not None and valid[u].any():
            values = fit[u, valid[u]]
            entry.update(fit_psnr_mean=float(values.mean()), fit_psnr_min=float(values.min()), fit_psnr_max=float(values.max()))
    a.update(z=z, zmean=means, zstd=stds, fit_psnr=fit)
    if 'valid' in a or not valid.all(): a['valid'] = valid
    m.update(checkpoint=str(resolve(CHECKPOINT)), checkpoint_sha256=checkpoint_hash,
             checkpoint_info=checkpoint_info, encoder_signature=encoder_hash, z_input_signature=signature,
             z_encoding_method='phase1.encode(original, edited), eval mode; EXIF/RGB/LANCZOS alignment',
             z_updated_at_utc=datetime.now(timezone.utc).isoformat(),
             valid_images_per_user={u: int(valid[i].sum()) for i, u in enumerate(users)})
    # 专家缓存沿用 train.py 的失效检查约定。
    if root == ROOT / 'data/raw':
        from train import expert_cache_context
        _, context = expert_cache_context(root)
        m.update(context)
        m['checkpoint_sha256'] = checkpoint_hash
        m['encoder_signature'] = encoder_hash
    tmp = root / '.z_refresh.tmp.npz'
    tmp_json = root / '.z_refresh.tmp.json'
    try:
        np.savez_compressed(tmp, **a)
        m['latent_sha256'] = sha(tmp)
        tmp_json.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(npz_path)
        tmp_json.replace(root / 'user_parameters.json')
    finally:
        tmp.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
    print(f'已覆盖原缓存：{npz_path}，z={z.shape}', flush=True)


def r_squared(centered, labels, counts, total):
    sums = np.column_stack([np.bincount(labels, weights=centered[:, j], minlength=len(counts)) for j in range(3)])
    return float(np.clip(np.sum(sums * sums / counts[:, None]) / total, 0, 1))


def analyze_user(z, labels, seed):
    z = np.asarray(z, np.float64)
    groups, labels = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    counts = np.bincount(labels)
    x = z.copy()
    if STANDARDIZE:
        scale = x.std(0)
        x = (x - x.mean(0)) / np.where(scale > 1e-12, scale, 1.)
    dist = cdist(x, x)
    np.fill_diagonal(dist, 0.)
    silhouette = (silhouette_samples(dist, labels, metric='precomputed')
                  if 1 < len(groups) < len(z) else np.full(len(z), np.nan))
    within, between, group_rows = [], [], []
    for g, group in enumerate(groups):
        mask = labels == g
        n = int(mask.sum())
        intra = dist[np.ix_(mask, mask)].sum(1) / (n - 1) if n > 1 else None
        inter = dist[np.ix_(mask, ~mask)].mean(1) if (~mask).any() else None
        # 总体按图片等权；单样本类别不参与距离比。
        if intra is not None and inter is not None:
            within.extend(intra); between.extend(inter)
        d_in, d_out = float(intra.mean()) if intra is not None else None, float(inter.mean()) if inter is not None else None
        row = dict(group=str(group), n=n, within_distance=d_in, between_distance=d_out,
                   distance_ratio=d_in / d_out if d_in is not None and d_out and d_out > 0 else None,
                   silhouette=float(silhouette[mask].mean()) if np.isfinite(silhouette[mask]).all() else None)
        row.update({f'z{j + 1}_mean': float(z[mask, j].mean()) for j in range(3)})
        row.update({f'z{j + 1}_std': float(z[mask, j].std()) for j in range(3)})
        group_rows.append(row)
    centered = x - x.mean(0)
    total = float(np.square(centered).sum())
    observed = p = null_mean = None
    if 1 < len(groups) < len(z) and total > 1e-20:
        observed = r_squared(centered, labels, counts, total)
        rng = np.random.default_rng(seed)
        null = np.asarray([r_squared(centered, rng.permutation(labels), counts, total) for _ in range(PERMUTATIONS)])
        p = float((1 + (null >= observed - 1e-12).sum()) / (PERMUTATIONS + 1))
        null_mean = float(null.mean())
    d_in = float(np.mean(within)) if within else None
    d_out = float(np.mean(between)) if between else None
    summary = dict(n=len(z), groups=len(groups), within_distance=d_in, between_distance=d_out,
                   distance_ratio=d_in / d_out if d_in is not None and d_out and d_out > 0 else None,
                   silhouette=float(silhouette.mean()) if np.isfinite(silhouette).all() else None,
                   r_squared=observed, permutation_null_r_squared=null_mean, permutation_p=p)
    return summary, group_rows, x


def holm(rows):
    ranked = sorted((r['permutation_p'], i) for i, r in enumerate(rows) if r['permutation_p'] is not None)
    previous = 0.
    for j, (p, i) in enumerate(ranked):
        previous = max(previous, min(1., (len(ranked) - j) * p))
        rows[i]['holm_p'] = previous
    for r in rows: r.setdefault('holm_p', None)


def plot_user(out, user, x, labels, names, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    groups = sorted(set(labels))
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(groups)))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, (i, j) in zip(axes[:3], ((0, 1), (0, 2), (1, 2))):
        for group, color in zip(groups, colors):
            mask = labels == group
            ax.scatter(x[mask, i], x[mask, j], s=7, alpha=.35, color=color, label=names.get(group, group), rasterized=True)
            ax.scatter(x[mask, i].mean(), x[mask, j].mean(), s=65, color=color, edgecolors='black', marker='X')
        ax.set(xlabel=f'z{i + 1}', ylabel=f'z{j + 1}')
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8, markerscale=2)
    positions = np.arange(len(rows))
    axes[3].barh(positions, [r['distance_ratio'] if r['distance_ratio'] is not None else np.nan for r in rows], color=colors)
    axes[3].axvline(1, color='black', linestyle='--', linewidth=1)
    axes[3].set(yticks=positions, yticklabels=[names.get(r['group'], r['group']) for r in rows], xlabel='类内 / 类间平均距离（越小越集中）')
    fig.suptitle(f'{user}：同类 z 分布（X 为类别中心；' + ('各维标准化' if STANDARDIZE else '原始 z') + '）')
    fig.tight_layout()
    fig.savefig(out / f'{user}.png', dpi=160)
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    import torch
    from phase1.model import Model
    if PERMUTATIONS < 1: raise ValueError('PERMUTATIONS 必须为正整数')
    root = resolve(GROUP_DIR)
    # 一次读取权重；即使外部继续训练，本次所有缓存也使用同一个模型快照。
    content = resolve(CHECKPOINT).read_bytes()
    checkpoint_hash = hashlib.sha256(content).hexdigest()
    ckpt = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
    del content
    if ckpt['dim'] != 3: raise ValueError('要求三维 phase1 权重')
    info = {k: ckpt.get(k) for k in ('dim', 'width', 'lut_size', 'epoch', 'val_psnr', 'dataset')}
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if DEVICE == 'auto' else DEVICE)
    model = Model(3, int(ckpt['width']), int(ckpt['lut_size'])).to(device).eval().requires_grad_(False)
    model.load_state_dict(ckpt['model'], strict=True)
    del ckpt
    encoder_hash = hashlib.sha256(''.join(sha(ROOT / 'phase1' / f) for f in ('model.py', 'config.py')).encode()).hexdigest()
    roots = {root}
    if REFRESH_ALL_CACHES:
        for p in (ROOT / 'data').rglob('user_latents.npz'):
            parts = p.relative_to(ROOT / 'data').parts
            if any(q.startswith('.') or '.previous-' in q or q == 'z_analysis' for q in parts): continue
            roots.add(p.parent)
    print(f'使用权重 epoch={info["epoch"]}, sha256={checkpoint_hash}; device={device}', flush=True)
    for cache_root in sorted(roots):
        refresh_cache(cache_root, model, device, checkpoint_hash, info, encoder_hash)
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    a, m = load_cache(root)
    names = {}
    for manifest in root.glob('group_*.json'):
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        for category in meta.get('categories', []): names[str(category['id'])] = category['name']
    out = root / 'z_analysis'
    out.mkdir(exist_ok=True)
    summaries, details = [], []
    for u, user in enumerate(a['users'].astype(str)):
        mask = np.isfinite(a['z'][u]).all(1)
        if 'valid' in a: mask &= a['valid'][u]
        if not mask.any(): print(f'{user} 无有效 z，跳过'); continue
        labels = a['image_groups'][mask].astype(str)
        summary, rows, x = analyze_user(a['z'][u, mask], labels, SEED + u)
        summary = dict(user=user, **summary, skipped=int((~mask).sum()))
        summaries.append(summary)
        details.extend(dict(user=user, name=names.get(r['group'], r['group']), **r) for r in rows)
        plot_user(out, user, x, labels, names, rows)
    if not summaries: raise ValueError('没有可分析的用户 z')
    holm(summaries)
    write_csv(out / 'summary.csv', summaries)
    write_csv(out / 'groups.csv', details)
    changed = sha(resolve(CHECKPOINT)) != checkpoint_hash
    report = dict(group_dir=str(root), checkpoint_sha256=checkpoint_hash, checkpoint_info=info,
                  checkpoint_changed_during_run=changed, standardized=STANDARDIZE,
                  permutations=PERMUTATIONS, seed=SEED, users=summaries, groups=details)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['同用户、同类别 z 相似性分析',
             '距离比 < 1：同类更近；轮廓系数约 0：类别重叠；R²：类别解释的 z 变异比例。',
             'p 是保持人数不变的标签置换检验；Holm p 校正了多个用户的检验。',
             '总体距离按图片等权，排除单样本类别；各类别结果见 groups.csv。',
             '统计关联不等于类别分离，也不能单独证明用户偏好独立于原图内容。', '']
    def fmt(v): return 'N/A' if v is None else f'{v:.4f}'
    for r in summaries:
        lines.append(f"{r['user']}: n={r['n']}, 类内/类间={fmt(r['distance_ratio'])}, silhouette={fmt(r['silhouette'])}, R²={fmt(r['r_squared'])}, Holm p={fmt(r['holm_p'])}")
    if changed: lines.append('运行期间权重文件已更新；本次所有 z 使用报告中记录的同一权重快照。再次运行可同步更新。')
    (out / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'结果：{out}', flush=True)


if __name__ == '__main__': main()
