# ========================= 分类设置 =========================
MODE = 3                         # 1=语义→data/semantic；2=色彩→data/color；3=三个随机网络分别编码→data/random1..3
RANDOM_SEEDS = (42, 43, 44)       # 模式 3：三个独立随机初始化 ResNet18；不同种子各保存一份权重
RANDOM_MODEL_DIR = "model/random_resnet18"  # 模式 3：首次初始化保存，以后复用；不加载预训练、不训练
RANDOM_IMAGE_SIZE = 224           # 模式 3：完整原图缩放尺寸，每个网络独立输出 512 维并独立聚类
RANDOM_CLUSTER_SEED = 42          # 模式 3：六类 KMeans 随机种子
RANDOM_REVIEW_MARGIN = 0.02       # 模式 3：前两中心的距离差小于此值时标记复核
COLOR_SEED = 42                  # 模式 2：KMeans 随机种子，相同输入和配置可复现
COLOR_WEIGHTS = (1., 1., 1., 1.)  # 模式 2：色温、亮度、对比度、色调的距离权重（标准化后）
COLOR_REVIEW_MARGIN = 0.05       # 模式 2：最近两中心的距离差低于此值时标记复核
MODEL = "openai/clip-vit-base-patch32"  # 模式 1：语义编码模型，模式 2 不加载模型
VIEW_WEIGHTS = (0.8, 0.2)          # 完整画面优先，中心裁剪辅助
REVIEW_MARGIN = 0.025             # 前两类分差低于此值，标记需要复核
MIN_SCORE = 0.22                  # 弱语义匹配复核阈值；不是概率
BOUNDARY_WEIGHT = 0.35            # 易混淆类别的额外对比权重
# ===========================================================
"""仅依据 O 原图分类；顶部 MODE 或 --mode 选择方式，三种方式独立保存，user.py 自动跟随最近成功运行。
1：语义分类，1人物、2动物、3植物、4风景、5城市、6室内。
2：色温/亮度/对比度/色调特征的距离聚类，1–6 为六个色彩组，没有固定语义名称。
3：三个固定随机 ResNet18 分别编码、分别按向量距离聚成六组，共 18 组。
三个网络不加载预训练、不训练；第一次保存到 model/random_resnet18，以后重复运行或 --recompute 均复用权重。
模式 3 分别输出 data/random1、data/random2、data/random3；每个目录均有 1–6、独立清单和 features.npz。
python group.py --mode 3 [--input data/raw]
python group.py --mode 2 [--input data/raw] [--output data/color]
色温为线性 RGB 均值的 (R-B)/(R+B) 冷暖近似，不是实测开尔文色温；亮度/对比度为线性亮度的均值/标准差。
色调为 HSV 环形角度，用色彩强度加权的 cos/sin 表示，因此四种属性对应五维距离向量。
三个标量分别标准化，色调两轴共用尺度；按 COLOR_WEIGHTS 加权后进行六类 KMeans。
模式 2 每次直接重算轻量色彩统计；features.npz 的 embeddings 为 N×5，无 L2 归一化，view_embeddings 为 N×0×5。

python group.py [--batch-size 16] [--device cpu] [--recompute] [--review]
python group.py --overrides data/semantic/review.csv

模式 1 依赖 torch、transformers、huggingface_hub、Pillow、numpy；模式 2 仅需 Pillow、numpy、scikit-learn；模式 3 另需 torch、torchvision，无需下载模型。
完整画面+中心视图；每类多种具体主体提示；人/动物、植物/风景、城市/室内另做对比。
六类按画面主旨区分，混合场景仍可能有歧义；分数和分差均不是校准概率。
review.csv 列出全部图片，优先排列待复核图片；填写 override_group（1–6 或中文类别名）后重跑。
--review 额外生成待复核图片的联系表；不增加“其他”类别，不强行平衡类别数量。
人工纠正存为 overrides.csv，同模式下默认沿用（色彩/随机模式还需特征与聚类配置一致）；切换模式不自动沿用旧纠正。原图改变时拒绝沿用旧纠正。

输入 data/raw/编号/O.png（data/row 存在时优先），缺少 O 的目录自动跳过。
模式 1 输出 data/semantic/1–6、group_semantic.json；模式 2 输出 data/color/1–6、group_color.json。
模式 3 输出 data/random1..3/1–6、group_random1..3.json；每套目录各自保留 features.npz、review.csv、overrides.csv，互不覆盖。
成功发布后更新 data/current_group.json，供 user.py 自动选择这三套目录并在每套生成一个用户。
重新运行同一方式时先把该方式的旧输出整体移到 目录名.previous-时间，再发布新分类；其他方式保持不动。
重新分组后需基于新分类重新生成模拟用户；不会悄悄重写旧用户的类别标签。
"""
import argparse, csv, hashlib, json, os, shutil, tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUP_LAYOUT = {1: ('semantic', 'group_semantic.json', '语义'),
                2: ('color', 'group_color.json', '色彩')}
RANDOM_LAYOUT = tuple((f'random{i}', f'group_random{i}.json', f'随机网络{i}', i - 1)
                      for i in range(1, len(RANDOM_SEEDS) + 1))


def default_output(mode):
    if mode == 3:
        raise ValueError('模式 3 有三个固定输出：data/random1、data/random2、data/random3')
    if mode not in GROUP_LAYOUT: raise ValueError(f'未知分组方式：{mode}')
    return ROOT / 'data' / GROUP_LAYOUT[mode][0]


def remember_group(output, mode):
    """只在分组完整发布后更新；user.py 使用实际运行结果，而不是尚未运行的 MODE 设置。"""
    output = Path(output).resolve()
    manifest = output / GROUP_LAYOUT[mode][1]
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    if metadata.get('mode', 1) != mode: raise ValueError('分组记录的模式与目录不符')
    data = ROOT / 'data'; data.mkdir(parents=True, exist_ok=True)
    state = dict(mode=mode, output=str(output), manifest=manifest.name, manifest_sha256=digest(manifest))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.current_group-', suffix='.tmp',
                                         dir=data, delete=False) as f:
            temporary = Path(f.name)
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(temporary, data/'current_group.json')
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def remember_random_groups(outputs):
    """模式 3 只在三个独立结果均已发布后，整体更新 user.py 的输入指针。"""
    if len(outputs) != len(RANDOM_LAYOUT):
        raise ValueError('随机网络输出数量不正确')
    items = []
    for (directory, manifest_name, _name, index), output in zip(RANDOM_LAYOUT, outputs):
        output = Path(output).resolve()
        manifest = output / manifest_name
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        if metadata.get('mode') != 3 or metadata.get('network_index') != index:
            raise ValueError(f'随机网络 {index + 1} 的分组记录不匹配')
        items.append(dict(output=str(output), manifest=manifest.name, manifest_sha256=digest(manifest),
                          network_index=index, directory=directory))
    data = ROOT / 'data'; data.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='.current_group-', suffix='.tmp',
                                         dir=data, delete=False) as f:
            temporary = Path(f.name)
            json.dump(dict(mode=3, outputs=items), f, ensure_ascii=False, indent=2)
        os.replace(temporary, data/'current_group.json')
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def current_group():
    """返回最近成功分组的目录、清单和来源信息；兼容尚未写入关联记录的旧分组。"""
    state_path = ROOT/'data/current_group.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else None
    mode = state['mode'] if state else MODE
    if mode == 3:
        raise ValueError('当前是模式 3 的三套随机分组，请由 user.py 通过 current_groups() 一起读取')
    if mode not in GROUP_LAYOUT: raise ValueError(f'不支持的分组方式：{mode}，请重新运行 group.py')
    root = Path(state['output']).resolve() if state else default_output(mode).resolve()
    manifest = root/GROUP_LAYOUT[mode][1]
    if not manifest.is_file(): raise FileNotFoundError(f'未找到分组结果 {manifest}，请先运行 group.py')
    digest_value = digest(manifest)
    if state and (state.get('manifest') != manifest.name or state.get('manifest_sha256') != digest_value):
        raise ValueError('当前分组记录与结果不一致，请重新运行 group.py，避免生成错误分组的用户')
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    if metadata.get('mode', 1) != mode: raise ValueError('分组模式不匹配，请重新运行 group.py')
    info = dict(mode=mode, name=GROUP_LAYOUT[mode][2], manifest=str(manifest), manifest_sha256=digest_value,
                method=metadata.get('method'), categories=metadata['categories'])
    return root, metadata, info


def current_groups():
    """返回最近发布的一套普通分组，或模式 3 的三套独立随机分组。"""
    state_path = ROOT/'data/current_group.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else None
    if not state or state.get('mode', MODE) != 3:
        return [current_group()]
    entries = state.get('outputs')
    if not isinstance(entries, list) or len(entries) != len(RANDOM_LAYOUT):
        raise ValueError('模式 3 的分组记录不是三个独立输出；请重新运行 group.py --mode 3')
    results = []
    for expected, entry in zip(RANDOM_LAYOUT, entries):
        directory, manifest_name, name, index = expected
        root = Path(entry.get('output', '')).resolve()
        manifest = root / manifest_name
        if (entry.get('directory') != directory or entry.get('manifest') != manifest_name
                or entry.get('network_index') != index or not manifest.is_file()
                or entry.get('manifest_sha256') != digest(manifest)):
            raise ValueError('随机分组记录与结果不一致，请重新运行 group.py --mode 3')
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        if metadata.get('mode') != 3 or metadata.get('network_index') != index:
            raise ValueError('随机分组清单不匹配，请重新运行 group.py --mode 3')
        results.append((root, metadata, dict(mode=3, name=name, manifest=str(manifest),
                                             manifest_sha256=digest(manifest), method=metadata.get('method'),
                                             categories=metadata['categories'], network_index=index,
                                             network_seed=metadata.get('network_seed'))))
    return results


ENCODING_VERSION = "clip-fullframe-pad-and-center-v2"
MODEL_FILES = ("config.json", "pytorch_model.bin", "preprocessor_config.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.json", "merges.txt")
CATEGORIES = [
    ("人物", (
        "a photograph with a human face or human portrait as the main subject",
        "a photograph focusing on a person posing for the camera",
        "a photograph focusing on a group of people together",
        "a photograph focusing on people working or doing an activity",
        "a photograph focusing on a person playing sports or performing",
        "a photograph focusing on a child or a baby",
        "a photograph focusing on people at a wedding or celebration",
        "a photograph focusing on a human body or a person in the foreground")),
    ("动物", (
        "a photograph with a dog or a cat as the main subject",
        "a photograph with a bird as the main subject",
        "a photograph focusing on a wild mammal",
        "a close-up photograph of an animal face or animal body",
        "a photograph focusing on fish or aquatic animals",
        "a photograph focusing on an insect or a butterfly",
        "a photograph focusing on a reptile or an amphibian",
        "a photograph focusing on horses, cows, sheep or other farm animals")),
    ("植物", (
        "a close-up botanical photograph focusing on a flower or blossoms",
        "a close-up photograph focusing on leaves and their texture",
        "a close-up photograph focusing on a plant stem or a branch",
        "a botanical photograph of a single tree as the main subject",
        "a close-up photograph focusing on grass, ferns or small plants",
        "a photograph of a potted plant as the main subject",
        "a close-up photograph focusing on a shrub or a bush",
        "a botanical detail photograph of buds, seeds or petals")),
    ("风景", (
        "a wide natural landscape photograph of mountains and valleys",
        "a wide natural landscape photograph of a beach and the ocean",
        "a wide natural landscape photograph of a lake and its surroundings",
        "a wide natural landscape photograph of a river or a waterfall",
        "a wide forest landscape with many trees and the surrounding environment",
        "a wide natural landscape photograph of fields, hills or a desert",
        "a wide natural landscape photograph of the sky and the horizon",
        "a wide natural landscape photograph of snow, ice or rocks")),
    ("城市", (
        "an outdoor photograph of building exteriors on a city street",
        "an outdoor photograph focusing on a building facade or exterior architecture",
        "an outdoor photograph of a bridge and its surroundings",
        "an outdoor photograph of a city skyline",
        "an outdoor photograph of a town square or an urban streetscape",
        "an outdoor photograph of the exterior of a castle, church or monument",
        "an outdoor photograph of houses or man-made structures in a village",
        "an outdoor streetscape photograph of roads, buildings and traffic")),
    ("室内", (
        "a photograph taken inside a room showing the interior space",
        "a photograph taken inside a hall showing interior walls and a ceiling",
        "an interior photograph of a kitchen or a dining room",
        "an interior photograph of a living room, bedroom and furniture",
        "an interior photograph of an office, library or museum",
        "a photograph taken inside a building showing a corridor or staircase",
        "an interior photograph of a restaurant, cafe or shop",
        "a photograph of an indoor furnished space viewed from inside")),
]
BOUNDARIES = [
    ((1, 2), (
        ("the main subject of this photograph is a human being", "this picture is about human faces, people and human activities"),
        ("the main subject of this photograph is an animal", "this picture is about an animal, a pet or wildlife"))),
    ((3, 4), (
        ("this picture focuses on the details of an individual plant, flower or leaves", "a botanical subject fills the frame in this plant photograph"),
        ("this picture shows a broad natural environment and scenery", "a landscape view of the surrounding terrain, forests, water and sky"))),
    ((5, 6), (
        ("this picture shows the exterior of buildings viewed from outdoors", "an outdoor architectural scene or city street seen in the open air"),
        ("this picture shows an interior room viewed from inside a building", "an enclosed interior space with walls, ceiling, floor and furniture"))),
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def find_originals(source):
    if not source.is_dir(): raise FileNotFoundError(f"输入目录不存在：{source}")
    folders = sorted((p for p in source.iterdir() if p.is_dir() and p.name.isdecimal()), key=lambda p:int(p.name))
    ids = [int(p.name) for p in folders]
    if len(set(ids)) != len(ids): raise ValueError("存在重复的数字图片编号")
    paths = [p/'O.png' for p in folders if (p/'O.png').is_file()]
    print(f"有效原图 {len(paths)}，缺少 O 跳过 {len(folders)-len(paths)}", flush=True)
    if not paths: raise ValueError(f"没有找到 编号/O.png：{source}")
    return paths


def load_clip(device_name):
    os.environ.setdefault('USE_TF', '0')
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model_dir = ROOT/'model'/MODEL.split('/')[-1]
    if not all((model_dir/name).is_file() for name in MODEL_FILES):
        from huggingface_hub import snapshot_download
        print(f"下载模型到 {model_dir}", flush=True)
        snapshot_download(MODEL, local_dir=str(model_dir), allow_patterns=list(MODEL_FILES))
    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
    device = torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True, use_safetensors=False).eval().to(device)
    signature = hashlib.sha256(json.dumps(dict(version=ENCODING_VERSION, model=MODEL,
                       files={name:digest(model_dir/name) for name in MODEL_FILES}), sort_keys=True).encode()).hexdigest()
    return model, processor, device, signature


def cached_features(output, originals, hashes, signature):
    import numpy as np
    try:
        manifest = json.loads((output/'group_semantic.json').read_text(encoding='utf-8'))
        with np.load(output/'features.npz', allow_pickle=False) as a:
            features = a['view_embeddings'].copy()
            if (manifest.get('encoding_signature') == signature and a['ids'].tolist() == [p.parent.name for p in originals]
                    and a['source_sha256'].tolist() == hashes and features.shape == (len(originals), 2, 512)
                    and np.isfinite(features).all() and (np.linalg.norm(features, axis=2) > 0).all()):
                print('复用已核对来源的双视图特征，重新计算类别分数', flush=True)
                return features
    except (OSError, KeyError, ValueError, TypeError): pass
    return None


def extract_features(originals, batch_size, model, processor, device):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    batches = []
    color = tuple(round(v*255) for v in processor.image_processor.image_mean)
    with torch.inference_mode():
        for start in range(0, len(originals), batch_size):
            views = []
            for path in originals[start:start+batch_size]:
                with Image.open(path) as im: image = ImageOps.exif_transpose(im).convert('RGB')
                side = max(image.size)
                full = Image.new('RGB', (side, side), color)
                full.paste(image, ((side-image.width)//2, (side-image.height)//2))
                # 第二张交给 CLIP 默认中心裁剪；第一张为完整画面，保留边缘场景信息。
                views.extend((full, image))
            pixels = processor(images=views, return_tensors='pt').to(device)
            features = F.normalize(model.get_image_features(**pixels).float(), dim=-1)
            batches.append(features.cpu().numpy().reshape(-1, 2, features.shape[-1]))
            print(f"提取双视图特征 {min(start+batch_size,len(originals))}/{len(originals)}", flush=True)
    return np.concatenate(batches)


def encode_prompts(model, processor, device):
    import torch
    import torch.nn.functional as F
    prompts = [p for _, descriptions in CATEGORIES for p in descriptions]
    prompts += [p for _, sides in BOUNDARIES for side in sides for p in side]
    with torch.inference_mode():
        tokens = processor(text=prompts, padding=True, truncation=True, return_tensors='pt').to(device)
        encoded = F.normalize(model.get_text_features(**tokens).float(), dim=-1)
    return encoded.cpu().numpy()


def classify(features, text):
    import numpy as np
    x, text = np.asarray(features, np.float32), np.asarray(text, np.float32)
    expected = sum(len(p) for _, p in CATEGORIES) + sum(len(p) for _, sides in BOUNDARIES for p in sides)
    if x.ndim!=3 or x.shape[1]!=2 or text.shape!=(expected,x.shape[-1]) or not len(x): raise ValueError('特征维度不正确')
    for a in (x,text):
        if not np.isfinite(a).all() or (np.linalg.norm(a,axis=-1)==0).any(): raise ValueError('特征包含无效值')
    x = x/np.linalg.norm(x,axis=-1,keepdims=True);text=text/np.linalg.norm(text,axis=-1,keepdims=True)
    similarities = np.clip(x@text.T,-1,1)
    per_view=[];offset=0;prototypes=[]
    for _, descriptions in CATEGORIES:
        n=len(descriptions)
        # 各类使用相同数量提示，以最匹配的两种具体描述计分，避免平均掉不同子类型。
        per_view.append(np.sort(similarities[:,:,offset:offset+n],axis=-1)[:,:,-2:].mean(-1))
        center=text[offset:offset+n].mean(0);prototypes.append(center/np.linalg.norm(center));offset+=n
    per_view=np.stack(per_view,axis=-1)
    weights=np.asarray(VIEW_WEIGHTS,np.float32);weights/=weights.sum()
    base=(per_view*weights[None,:,None]).sum(1);scores=base.copy()
    initial_rank=np.argsort(-base,axis=1,kind='stable')
    boundary_conflict=np.zeros(len(x),bool);boundary_used=['']*len(x)
    for (left,right),sides in BOUNDARIES:
        evidence=[]
        for side in sides:
            n=len(side)
            evidence.append((similarities[:,:,offset:offset+n].mean(-1)*weights).sum(1));offset+=n
        a,b=left-1,right-1
        eligible=np.isin(initial_rank[:,0],[a,b]) & np.isin(initial_rank[:,1],[a,b])
        delta=evidence[0]-evidence[1]
        scores[eligible,a]+=BOUNDARY_WEIGHT*delta[eligible]/2
        scores[eligible,b]-=BOUNDARY_WEIGHT*delta[eligible]/2
        boundary_conflict[eligible]=(np.where(delta>=0,a,b)[eligible]!=initial_rank[eligible,0])
        for i in np.flatnonzero(eligible): boundary_used[i]=f'{left}/{right}'
    rank=np.argsort(-scores,axis=1,kind='stable')
    margins=scores[np.arange(len(x)),rank[:,0]]-scores[np.arange(len(x)),rank[:,1]]
    reasons=[]
    for i in range(len(x)):
        row=[]
        if margins[i]<REVIEW_MARGIN: row.append('low_margin')
        if per_view[i,0].argmax()!=per_view[i,1].argmax(): row.append('full_crop_disagree')
        if boundary_conflict[i]: row.append('boundary_disagree')
        if base[i].max()<MIN_SCORE: row.append('weak_match')
        reasons.append(row)
    return dict(groups=rank[:,0]+1, scores=scores, margins=margins, rank=rank,
                per_view=per_view, base=base, reasons=reasons, boundary_used=boundary_used,
                class_embeddings=np.stack(prototypes))


def read_overrides(path, originals, hashes, categories=None, mode=1):
    if not path or not path.is_file(): return {}
    lookup={int(p.parent.name):h for p,h in zip(originals,hashes)}
    categories = CATEGORIES if categories is None else categories
    names={name:str(i+1) for i,(name,_) in enumerate(categories)}
    overrides={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or 'id' not in reader.fieldnames or not {'override_group','group'}&set(reader.fieldnames):
            raise ValueError(f'纠正表需要 id 和 override_group/group 列：{path}')
        for row in reader:
            value=((row.get('override_group') if 'override_group' in row else row.get('group','')) or '').strip()
            if not value:continue
            if row.get('mode') and int(row['mode']) != mode: raise ValueError(f'纠正表的分类模式与当前模式不同：{path}')
            image_id=int(row['id']);value=names.get(value,value)
            if value not in [str(i+1) for i in range(len(categories))]: raise ValueError(f'无效纠正类别：{image_id}: {value}')
            if image_id not in lookup:raise ValueError(f'纠正编号不在当前输入中：{image_id}')
            if row.get('source_sha256') and row['source_sha256']!=lookup[image_id]:raise ValueError(f'原图已改变，请重新复核：{image_id}')
            if image_id in overrides and overrides[image_id]!=int(value):raise ValueError(f'重复且冲突的纠正：{image_id}')
            overrides[image_id]=int(value)
    return overrides


def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def review_sheets(stage, records):
    from PIL import Image,ImageOps,ImageDraw,ImageFont
    pending=sorted((r for r in records if r['needs_review']),key=lambda r:r['margin'])
    if not pending:return
    folder=stage/'review';folder.mkdir()
    try:font=ImageFont.truetype('msyh.ttc',13)
    except OSError:font=ImageFont.load_default()
    for start in range(0,len(pending),25):
        page=pending[start:start+25]
        canvas=Image.new('RGB',(900,190*((len(page)+4)//5)),(245,245,245));draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(page):
            x=(j%5)*180;y=(j//5)*190
            with Image.open(row['source']) as im:thumb=ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'),(170,132))
            canvas.paste(thumb,(x+(180-thumb.width)//2,y+(132-thumb.height)//2))
            draw.text((x+4,y+134),f"{row['id']}  {row['group']} {row['category']}",font=font,fill=(20,20,20))
            alternative=row['top_categories'][1]
            draw.text((x+4,y+152),f"备选 {alternative['id']} {alternative['name']}",font=font,fill=(130,45,30))
            draw.text((x+4,y+170),f"margin={row['margin']:.3f}",font=font,fill=(90,90,90))
        canvas.save(folder/f'{start//25+1:03d}.jpg',quality=92)


def color_features(originals):
    """四项描述值 + 五维距离向量；色调用带色彩强度的 cos/sin 表示。"""
    import numpy as np
    from PIL import Image, ImageOps
    descriptions, vectors = [], []
    for i, path in enumerate(originals):
        with Image.open(path) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'), np.float64) / 255.
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        red, _, blue = linear.mean(axis=(0, 1))
        warmth = (red - blue) / (red + blue + 1e-8)
        luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
        brightness, contrast = float(luminance.mean()), float(luminance.std())
        maximum, minimum = rgb.max(2), rgb.min(2)
        chroma = maximum - minimum
        denominator = np.where(chroma > 1e-8, chroma, 1.)
        r, g, b = np.moveaxis(rgb, -1, 0)
        hue = np.select([maximum == r, maximum == g],
                        [(g-b)/denominator, (b-r)/denominator+2], default=(r-g)/denominator+4)
        angle = (hue / 6. % 1.) * (2 * np.pi)
        # chroma=S*V：灰色、黑色不贡献任意色调；多种相反色调会互相抵消。
        hx, hy = float((chroma*np.cos(angle)).mean()), float((chroma*np.sin(angle)).mean())
        hue_degrees = float(np.degrees(np.arctan2(hy, hx)) % 360) if np.hypot(hx, hy) > 1e-8 else 0.
        descriptions.append([warmth, brightness, contrast, hue_degrees])
        vectors.append([warmth, brightness, contrast, hx, hy])
        if (i+1) % 200 == 0 or i+1 == len(originals):
            print(f'提取色彩特征 {i+1}/{len(originals)}', flush=True)
    return np.asarray(descriptions, np.float64), np.asarray(vectors, np.float64)


def classify_color(originals):
    import numpy as np
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    weights = np.asarray(COLOR_WEIGHTS, np.float64)
    if weights.shape != (4,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('COLOR_WEIGHTS 必须是四个正数，依次为色温、亮度、对比度、色调的权重')
    if not np.isfinite(COLOR_REVIEW_MARGIN) or COLOR_REVIEW_MARGIN < 0:
        raise ValueError('COLOR_REVIEW_MARGIN 必须是非负有限数')
    if len(originals) < 6: raise ValueError('色彩聚类需要至少 6 张有效 O 原图')
    descriptions, vectors = color_features(originals)
    mean, scale = vectors.mean(0), vectors.std(0)
    # 色调两轴共用一个尺度，联合方差为 1，既保持圆周距离，也不重复计算色调权重。
    scale[3:] = np.sqrt(np.square(vectors[:, 3:]-mean[3:]).sum(1).mean())
    scale[scale < 1e-6] = 1.
    effective_weights = np.sqrt(weights[[0, 1, 2, 3, 3]])
    embeddings = (vectors-mean) / scale * effective_weights
    if len(np.unique(embeddings, axis=0)) < 6:
        raise ValueError('不同色彩向量不足 6 个，无法生成六个非空类别；请增加不同原图')
    with threadpool_limits(limits=1):
        estimator = KMeans(n_clusters=6, n_init=20, random_state=COLOR_SEED).fit(embeddings)
    centers = estimator.cluster_centers_
    # 冷到暖排序，以亮度、对比度、色调坐标破除并列，保证同数据/配置可复现。
    order = np.lexsort(tuple(centers[:, j] for j in range(4, -1, -1)))
    centers = centers[order]
    distances = np.sqrt(np.square(embeddings[:, None]-centers[None]).sum(2))
    rank = np.argsort(distances, axis=1, kind='stable')
    margins = distances[np.arange(len(rank)), rank[:, 1]]-distances[np.arange(len(rank)), rank[:, 0]]
    groups = rank[:, 0]+1
    if len(np.unique(groups)) != 6: raise ValueError('聚类未形成六个非空类别，请检查重复图片/色彩特征')
    settings = dict(seed=COLOR_SEED, n_init=20, color_weights=weights.tolist(), review_margin=COLOR_REVIEW_MARGIN,
                    feature_names=['warmth_red_blue', 'mean_linear_luminance', 'std_linear_luminance', 'hue_degrees'],
                    distance_feature_names=['warmth_red_blue', 'mean_linear_luminance', 'std_linear_luminance',
                                            'chroma_weighted_hue_cos', 'chroma_weighted_hue_sin'],
                    feature_mean=mean.tolist(), feature_scale=scale.tolist(), metric='weighted standardized Euclidean')
    signature = hashlib.sha256(json.dumps(dict(version='color-statistics-circular-hue-v1', settings=settings),
                                         sort_keys=True).encode()).hexdigest()
    result = dict(groups=groups, scores=-distances, base=-distances, margins=margins, rank=rank,
                  reasons=[['low_margin'] if m < COLOR_REVIEW_MARGIN else [] for m in margins],
                  boundary_used=['']*len(groups), class_embeddings=centers)
    arrays = dict(embeddings=embeddings.astype(np.float32),
                  view_embeddings=np.empty((len(groups), 0, 5), np.float32),  # 色彩模式无 CLIP 双视图
                  color_features=descriptions, color_vectors=vectors, feature_mean=mean, feature_scale=scale,
                  feature_names=np.asarray(settings['feature_names']),
                  distance_feature_names=np.asarray(settings['distance_feature_names']))
    return result, arrays, settings, signature


def load_random_networks(device_name):
    """三个不同 seed 的 ResNet18，weights=None；首次保存，以后只加载原权重。"""
    import torch
    import torchvision
    from torch import nn
    from torchvision.models import resnet18
    if len(RANDOM_SEEDS) != 3 or len(set(RANDOM_SEEDS)) != 3:
        raise ValueError('RANDOM_SEEDS 必须包含三个不同的整数种子')
    if any(type(seed) is not int or seed < 0 or seed >= 2**63 for seed in RANDOM_SEEDS):
        raise ValueError('随机网络种子必须为 [0, 2**63) 的整数')
    if RANDOM_IMAGE_SIZE < 32: raise ValueError('RANDOM_IMAGE_SIZE 至少为 32')
    device = torch.device(device_name or ('cuda' if torch.cuda.is_available() else 'cpu'))
    folder = Path(RANDOM_MODEL_DIR)
    folder = (folder if folder.is_absolute() else ROOT/folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    models, metadata = [], []
    for seed in RANDOM_SEEDS:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            net = resnet18(weights=None)
            net.fc = nn.Identity()
        path = folder/f'resnet18_seed{seed}.pt'
        if path.exists():
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            if checkpoint.get('architecture') != 'random_resnet18_pool512_v1' or checkpoint.get('seed') != seed:
                raise ValueError(f'随机网络权重配置不匹配：{path}；请使用独立模型目录')
            net.load_state_dict(checkpoint['model'], strict=True)
            print(f'复用随机网络：{path}', flush=True)
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix='.random-', suffix='.pt', dir=folder, delete=False) as f:
                    temporary = Path(f.name)
                torch.save(dict(architecture='random_resnet18_pool512_v1', seed=seed, pretrained=False,
                                torchvision_version=str(torchvision.__version__), model=net.state_dict()), temporary)
                os.replace(temporary, path)
            finally:
                if temporary is not None and temporary.exists(): temporary.unlink()
            print(f'初始化并保存随机网络：{path}', flush=True)
        models.append(net.eval().requires_grad_(False).to(device))
        metadata.append(dict(seed=seed, path=str(path), sha256=digest(path)))
    settings = dict(architecture='random_resnet18_pool512_v1', pretrained=False, trained=False,
                    networks=metadata, image_size=RANDOM_IMAGE_SIZE, per_network_dim=512,
                    preprocess='EXIF RGB; resize full image bilinear antialias; ImageNet fixed mean/std',
                    normalization='L2 each network independently',
                    metric='Each network is clustered independently; embeddings are never concatenated',
                    cluster_seed=RANDOM_CLUSTER_SEED, n_init=20, review_margin=RANDOM_REVIEW_MARGIN)
    signature = hashlib.sha256(json.dumps(dict(version='three-random-resnet18-v1', settings=settings,
                                              torch=str(torch.__version__), torchvision=str(torchvision.__version__)),
                                         sort_keys=True).encode()).hexdigest()
    return models, device, settings, signature


def random_network_features(originals, batch_size, models, device):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from torchvision.transforms import functional as TF, InterpolationMode
    batches = []
    with torch.inference_mode():
        for start in range(0, len(originals), batch_size):
            images = []
            for path in originals[start:start+batch_size]:
                with Image.open(path) as im: rgb = ImageOps.exif_transpose(im).convert('RGB')
                rgb = TF.resize(rgb, [RANDOM_IMAGE_SIZE, RANDOM_IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR,
                                antialias=True)
                images.append(TF.normalize(TF.to_tensor(rgb), (0.485,0.456,0.406), (0.229,0.224,0.225)))
            inputs = torch.stack(images).to(device)
            encoded = torch.stack([net(inputs).float() for net in models], dim=1)
            norms = encoded.norm(dim=2, keepdim=True)
            if not torch.isfinite(encoded).all() or (norms < 1e-12).any():
                raise ValueError('随机网络产生了非有限或零向量')
            batches.append((encoded/norms).cpu().numpy())
            print(f'提取三个随机网络特征 {min(start+batch_size,len(originals))}/{len(originals)}', flush=True)
    return np.concatenate(batches)


def classify_random_networks(originals, batch_size, device_name):
    """三个随机网络各自编码、各自 KMeans；绝不拼接后只聚成一套。"""
    import numpy as np
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    if len(originals) < 6: raise ValueError('随机网络聚类需要至少 6 张有效 O 原图')
    if not np.isfinite(RANDOM_REVIEW_MARGIN) or RANDOM_REVIEW_MARGIN < 0:
        raise ValueError('RANDOM_REVIEW_MARGIN 必须是非负有限数')
    models, device, all_settings, _all_signature = load_random_networks(device_name)
    features = random_network_features(originals, batch_size, models, device)
    del models
    results = []
    for directory, _manifest_name, name, index in RANDOM_LAYOUT:
        embeddings = features[:, index, :].astype(np.float64, copy=True)
        if len(np.unique(np.round(embeddings, 6), axis=0)) < 6:
            raise ValueError(f'{name} 的不同编码不足 6 个，无法生成六个非空组')
        with threadpool_limits(limits=1):
            estimator = KMeans(n_clusters=6, n_init=20, random_state=RANDOM_CLUSTER_SEED + index).fit(embeddings)
        centers = estimator.cluster_centers_
        order = np.lexsort(tuple(centers[:, j] for j in range(centers.shape[1]-1, -1, -1)))
        centers = centers[order]
        distances = np.stack([np.linalg.norm(embeddings-c, axis=1) for c in centers], axis=1)
        rank = np.argsort(distances, axis=1, kind='stable')
        margins = distances[np.arange(len(rank)), rank[:, 1]] - distances[np.arange(len(rank)), rank[:, 0]]
        groups = rank[:, 0] + 1
        if len(np.unique(groups)) != 6:
            raise ValueError(f'{name} 未形成六个非空组，请检查重复图片')
        network = all_settings['networks'][index]
        settings = dict(architecture=all_settings['architecture'], pretrained=False, trained=False,
                        network=network, network_index=index, image_size=RANDOM_IMAGE_SIZE,
                        embedding_dim=512, preprocess=all_settings['preprocess'], normalization='L2 per image',
                        metric='Euclidean KMeans on one random-network unit embedding',
                        cluster_seed=RANDOM_CLUSTER_SEED + index, n_init=20, review_margin=RANDOM_REVIEW_MARGIN)
        signature = hashlib.sha256(json.dumps(dict(version='one-random-resnet18-v2', settings=settings),
                                                  sort_keys=True).encode()).hexdigest()
        result = dict(groups=groups, scores=-distances, base=-distances, margins=margins, rank=rank,
                      reasons=[['low_margin'] if m < RANDOM_REVIEW_MARGIN else [] for m in margins],
                      boundary_used=[''] * len(groups), class_embeddings=centers)
        arrays = dict(embeddings=embeddings.astype(np.float32), network_embedding=embeddings.astype(np.float32),
                      network_index=np.asarray([index], np.int64), network_seed=np.asarray([network['seed']], np.int64),
                      view_embeddings=np.empty((len(groups), 0, 512), np.float32))
        results.append((directory, index, result, arrays, settings, signature))
    return results


def validate_output(output, source, manifest_name):
    output = Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('输入和输出不能互相包含')
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError('输出必须是普通目录')
    if output.exists() and any(output.iterdir()) and not (output/manifest_name).is_file():
        raise ValueError(f'输出目录不是本程序的对应分类结果：{output}')
    return output


def publish_result(mode, output, manifest_name, display_name, originals, hashes, result, feature_arrays, settings,
                   signature, model_name, method, note, args, network_index=None):
    import numpy as np
    output = validate_output(output, Path(args.input).resolve(), manifest_name)
    previous = json.loads((output/manifest_name).read_text(encoding='utf-8')) if (output/manifest_name).is_file() else {}
    if previous and previous.get('mode', 1) != mode:
        raise ValueError('输出目录属于另一种分组规则，拒绝覆盖')
    names = CATEGORIES if mode == 1 else [(f'{display_name}组{i}', ()) for i in range(1, 7)]
    compatible = previous.get('mode', 1) == mode and (mode == 1 or previous.get('encoding_signature') == signature)
    overrides = read_overrides(output/'overrides.csv', originals, hashes, names, mode) if compatible else {}
    if previous and not compatible:
        print(f'{display_name} 的特征或聚类配置改变，本次不自动沿用旧人工纠正', flush=True)
    if args.overrides:
        if not args.overrides.is_file(): raise FileNotFoundError(args.overrides)
        overrides.update(read_overrides(args.overrides, originals, hashes, names, mode))
    records = []
    for i, path in enumerate(originals):
        auto = int(result['groups'][i]); group = overrides.get(int(path.parent.name), auto)
        manual = int(path.parent.name) in overrides
        records.append(dict(id=path.parent.name, source=str(path), source_sha256=hashes[i], group=group,
                            category=names[group-1][0], file=f'{group}/{path.parent.name}.png', auto_group=auto,
                            decision='manual' if manual else 'automatic', score=float(result['scores'][i, group-1]),
                            margin=float(result['margins'][i]), needs_review=bool(result['reasons'][i]) and not manual,
                            review_reasons=result['reasons'][i], boundary_comparison=result['boundary_used'][i],
                            full_frame_group=int(result['per_view'][i, 0].argmax()+1) if mode == 1 else auto,
                            center_crop_group=int(result['per_view'][i, 1].argmax()+1) if mode == 1 else None,
                            top_categories=[dict(id=int(j+1), name=names[j][0], score=float(result['scores'][i, j]))
                                            for j in result['rank'][i, :3]]))
    groups = np.asarray([r['group'] for r in records], np.int64)
    categories = [dict(id=i+1, name=name, count=int((groups == i+1).sum()), prompts=list(prompts))
                  for i, (name, prompts) in enumerate(names)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        stage = Path(temporary)
        for category in categories: (stage/str(category['id'])).mkdir()
        for path, row in zip(originals, records):
            shutil.copy2(path, stage/row['file'])
            if digest(stage/row['file']) != row['source_sha256']:
                raise RuntimeError(f'分类期间原图改变：{path}')
        np.savez_compressed(stage/'features.npz', ids=np.asarray([r['id'] for r in records]),
                            source_sha256=np.asarray(hashes), **feature_arrays,
                            class_embeddings=result['class_embeddings'], similarities=result['scores'],
                            base_similarities=result['base'], groups=groups, margins=result['margins'])
        manifest = dict(mode=mode, model=model_name, method=method, encoding_signature=signature,
                        input=str(Path(args.input).resolve()), total=len(records), n_groups=len(names),
                        categories=categories, images=records, settings=settings,
                        boundary_prompts=BOUNDARIES if mode == 1 else [],
                        note=note+' 人工纠正优先，needs_review 不代表已确认错误。')
        if network_index is not None:
            manifest.update(network_index=network_index, network_seed=settings['network']['seed'])
        (stage/manifest_name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        review = [dict(id=r['id'], mode=mode, predicted_group=r['group'], category=r['category'],
                       alternative_group=r['top_categories'][1]['id'], margin=r['margin'],
                       needs_review=r['needs_review'], reasons=';'.join(r['review_reasons']), override_group='',
                       source_sha256=r['source_sha256'], source=r['source'])
                  for r in sorted(records, key=lambda r: (not r['needs_review'], r['margin']))]
        write_csv(stage/'review.csv', review, list(review[0]))
        corrected = [dict(id=r['id'], mode=mode, group=r['group'], source_sha256=r['source_sha256'])
                     for r in records if r['decision'] == 'manual']
        write_csv(stage/'overrides.csv', corrected, ['id', 'mode', 'group', 'source_sha256'])
        if args.review: review_sheets(stage, records)
        backup = None
        if output.exists():
            backup = output.with_name(output.name+'.previous-'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            output.rename(backup)
        try: stage.rename(output)
        except OSError:
            if backup and not output.exists(): backup.rename(output)
            raise
    print(f'完成 {display_name}：{len(records)} 张，6 类，待复核 {sum(r["needs_review"] for r in records)}，人工纠正 {len(overrides)}；{output}', flush=True)
    for category in categories: print(f'{category["id"]} {category["name"]}: {category["count"]} 张')
    if backup: print(f'旧分组及其中用户结果已保留于 {backup}')
    return output


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=ROOT/'data'/('row' if (ROOT/'data/row').is_dir() else 'raw'))
    parser.add_argument('--output', type=Path, default=None, help='模式 1/2 的自定义输出；模式 3 固定输出 random1..3')
    parser.add_argument('--mode', type=int, choices=(1, 2, 3), default=MODE, help='1=语义；2=色彩；3=三个随机网络分别聚类')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default=None)
    parser.add_argument('--recompute', action='store_true', help='模式 1 忽略既有 CLIP 特征；模式 3 每次重新编码')
    parser.add_argument('--overrides', type=Path, default=None)
    parser.add_argument('--review', action='store_true', help='额外导出低置信度图片复核拼图')
    args = parser.parse_args()
    if args.batch_size < 1: parser.error('batch-size 至少为 1')
    if args.mode == 3 and args.output is not None: parser.error('模式 3 固定写入 data/random1..3，不支持 --output')
    if args.mode == 3 and args.overrides is not None: parser.error('模式 3 请分别编辑 random1..3/overrides.csv 后重跑')
    source = args.input.resolve(); originals = find_originals(source); hashes = [digest(p) for p in originals]
    if args.mode == 1:
        output = args.output.resolve() if args.output else default_output(1).resolve()
        model, processor, device, signature = load_clip(args.device)
        features = None if args.recompute else cached_features(output, originals, hashes, signature)
        if features is None: features = extract_features(originals, args.batch_size, model, processor, device)
        result = classify(features, encode_prompts(model, processor, device))
        weights = np.asarray(VIEW_WEIGHTS, np.float32); weights /= weights.sum()
        embedding = (features*weights[None, :, None]).sum(1); embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        publish_result(1, output, 'group_semantic.json', '语义', originals, hashes, result,
                       dict(embeddings=embedding, view_embeddings=features),
                       dict(view_weights=VIEW_WEIGHTS, review_margin=REVIEW_MARGIN, min_score=MIN_SCORE, boundary_weight=BOUNDARY_WEIGHT),
                       signature, MODEL, 'full-frame/crop fine-grained CLIP scoring with pairwise boundary checks',
                       'score 是提示匹配与边界对比的启发式分数，不是概率；margin 是自动前两名分差。', args)
        remember_group(output, 1)
    elif args.mode == 2:
        output = args.output.resolve() if args.output else default_output(2).resolve()
        result, arrays, settings, signature = classify_color(originals)
        publish_result(2, output, 'group_color.json', '色彩', originals, hashes, result, arrays, settings, signature,
                       'image-color-statistics', 'KMeans on standardized warmth/brightness/contrast/circular hue',
                       'score 是到自动聚类中心的负欧氏距离，不是概率；margin 是次近距离减最近距离；编号仅表示色彩组。', args)
        remember_group(output, 2)
    else:
        published = []
        for (directory, manifest_name, name, index), (_dir, result_index, result, arrays, settings, signature) in zip(RANDOM_LAYOUT, classify_random_networks(originals, args.batch_size, args.device)):
            if directory != _dir or index != result_index: raise RuntimeError('随机网络输出顺序异常')
            output = ROOT/'data'/directory
            published.append(publish_result(3, output, manifest_name, name, originals, hashes, result, arrays, settings,
                                            signature, 'random-resnet18', 'KMeans on one L2-normalized random ResNet18 embedding',
                                            '固定随机初始化且未训练的 ResNet18；不使用预训练权重。score 是到自动聚类中心的负欧氏距离，编号仅表示随机特征簇。',
                                            args, network_index=index))
        remember_random_groups(published)
        print('模式 3 已完成：random1、random2、random3 各六组（共 18 组）。直接运行 user.py 会在每套内各生成一个用户。', flush=True)


if __name__ == '__main__': main()
