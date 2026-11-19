"""仅依据 O 原图，按 1人物、2动物、3植物、4风景、5城市、6室内分类。

运行：python group.py [--batch-size 16] [--device cpu] [--recompute]

依赖：pip install torch transformers huggingface_hub pillow numpy

输入：data/raw/编号/O.png（存在 data/row 时优先使用 row）。

输出：data/semantic/1–6/编号.png、features.npz、group_semantic.json。

只复制 O 原文件，数量不强制均衡；低分差样本标记 needs_review，仍保留在六类中。

首次运行下载模型到 model/clip-vit-base-patch32；旧结果会备份到 semantic.previous-时间。

模型：https://huggingface.co/openai/clip-vit-base-patch32

"""

import argparse

import hashlib

import json

import os

import shutil

import tempfile

from datetime import datetime

from pathlib import Path



ROOT = Path(__file__).resolve().parent

MODEL = "openai/clip-vit-base-patch32"

CATEGORIES = [

    ("人物", ("a portrait of a person", "people at a wedding or social gathering",

              "people doing everyday activities", "people playing sports or performing")),

    ("动物", ("an animal in its habitat", "a pet dog or cat",

              "a bird or a flock of birds", "an insect, fish or other wildlife")),

    ("植物", ("a close-up of flowers", "green leaves and plant foliage",

              "a tree or shrub as the main subject", "plants growing in a garden")),

    ("风景", ("a natural mountain or lake landscape", "a forest or waterfall landscape",

              "a beach, ocean or desert landscape", "a wide countryside landscape with fields and sky")),

    ("城市", ("an urban street with buildings", "a city skyline",

              "outdoor architecture, buildings or a bridge", "a town square or a street with traffic")),

    ("室内", ("the interior of a room", "an indoor hall, library or museum",

              "a kitchen, bedroom or living room interior", "furniture and objects in an indoor space")),

]

MODEL_FILES = ("config.json", "pytorch_model.bin", "preprocessor_config.json",

               "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")





def find_originals(source):

    if not source.is_dir():

        raise FileNotFoundError(f"输入目录不存在：{source}")

    originals = [p / "O.png" for p in sorted(source.iterdir()) if p.is_dir() and p.name.isdecimal() and (p / "O.png").is_file()]
    if not originals:
        raise ValueError(f"没有找到 编号/O.png：{source}")
    return originals





def digest(path):

    return hashlib.sha256(path.read_bytes()).hexdigest()





def cached_features(output, originals, hashes):

    import numpy as np

    try:

        manifest = json.loads((output / "group_semantic.json").read_text(encoding="utf-8"))

        with np.load(output / "features.npz", allow_pickle=False) as data:

            if manifest["model"] != MODEL or data["ids"].tolist() != [p.parent.name for p in originals]:

                return None

            if "source_sha256" in data:

                valid = data["source_sha256"].tolist() == hashes

            else:

                rows = {row["id"]: row for row in manifest["images"]}

                copies = [(output / rows[p.parent.name]["file"]).resolve() for p in originals]

                valid = all(output in p.parents and p.is_file() and digest(p) == h

                            for p, h in zip(copies, hashes))

            features = data["embeddings"].copy()

            if valid and features.shape == (len(originals), 512):

                print("复用已核对来源的 CLIP 图像特征，重新计算六类分数", flush=True)

                return features

    except (OSError, ValueError, KeyError, TypeError):

        pass

    return None





def load_clip(device_name):

    os.environ.setdefault("USE_TF", "0")

    import torch

    from huggingface_hub import snapshot_download

    from transformers import CLIPModel, CLIPProcessor

    model_dir = ROOT / "model" / MODEL.split("/")[-1]

    if not all((model_dir / name).is_file() for name in MODEL_FILES):

        print(f"下载模型到 {model_dir}", flush=True)

        snapshot_download(MODEL, local_dir=str(model_dir), allow_patterns=list(MODEL_FILES))

    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True, use_safetensors=False).eval().to(device)

    return model, processor, device





def encode_categories(model, processor, device):

    import torch

    import torch.nn.functional as F

    prompts = [f"a photo of {text}." for _, descriptions in CATEGORIES for text in descriptions]

    with torch.inference_mode():

        tokens = processor(text=prompts, padding=True, return_tensors="pt").to(device)

        text = F.normalize(model.get_text_features(tokens).float(), dim=-1)

        parts = text.split([len(descriptions) for _, descriptions in CATEGORIES])

        classes = F.normalize(torch.stack([part.mean(0) for part in parts]), dim=-1)

    return classes.cpu().numpy()





def extract_features(originals, batch_size, model, processor, device):

    import numpy as np

    import torch

    import torch.nn.functional as F

    from PIL import Image, ImageOps

    batches = []

    with torch.inference_mode():

        for start in range(0, len(originals), batch_size):

            images = []

            for path in originals[start:start + batch_size]:

                with Image.open(path) as image:

                    images.append(ImageOps.exif_transpose(image).convert("RGB"))

            pixels = processor(images=images, return_tensors="pt").to(device)

            features = F.normalize(model.get_image_features(pixels).float(), dim=-1)

            batches.append(features.cpu().numpy())

            print(f"提取特征 {min(start + batch_size, len(originals))}/{len(originals)}", flush=True)

    return np.concatenate(batches)





def classify(features, classes):

    import numpy as np

    vectors = [np.asarray(v, dtype=np.float32) for v in (features, classes)]

    for v in vectors:

        if v.ndim != 2 or not len(v) or not np.isfinite(v).all() or np.any(np.linalg.norm(v, axis=1) == 0):

            raise ValueError("特征必须是非空、有限且无零向量的二维数组")

    x, text = [v / np.linalg.norm(v, axis=1, keepdims=True) for v in vectors]

    if text.shape[0] != len(CATEGORIES) or x.shape[1] != text.shape[1]:

        raise ValueError("类别数量或特征维度不匹配")

    scores = np.clip(x @ text.T, -1, 1)

    ranked = np.sort(scores, axis=1)

    return scores.argmax(axis=1) + 1, scores, ranked[:, -1] - ranked[:, -2]





def main():

    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    default_input = ROOT / "data" / ("row" if (ROOT / "data/row").is_dir() else "raw")

    parser.add_argument("--input", type=Path, default=default_input)

    parser.add_argument("--output", type=Path, default=ROOT / "data/semantic")

    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--device", default=None, help="cuda、cuda:0 或 cpu；默认自动选择")

    parser.add_argument("--recompute", action="store_true", help="重新提取图像特征，不复用缓存")

    args = parser.parse_args()

    source, output = args.input.resolve(), args.output.resolve()

    if args.batch_size < 1:

        parser.error("batch-size 至少为 1")

    if output == source or source in output.parents or output in source.parents:

        raise ValueError("输入和输出目录不能互相包含")

    if args.output.is_symlink() or (output.exists() and not output.is_dir()):

        raise ValueError("输出路径必须是普通目录")

    if output.exists() and any(output.iterdir()) and not (output / "group_semantic.json").is_file():

        raise ValueError("输出目录包含非本脚本结果，请用 --output 指定新目录")

    originals = find_originals(source)

    hashes = [digest(p) for p in originals]

    features = None if args.recompute else cached_features(output, originals, hashes)

    model, processor, device = load_clip(args.device)

    if features is None:

        features = extract_features(originals, args.batch_size, model, processor, device)

    classes = encode_categories(model, processor, device)

    groups, scores, margins = classify(features, classes)

    records = [{"id": p.parent.name, "source": str(p), "group": int(gid),

                "category": CATEGORIES[gid - 1][0], "file": f"{gid}/{p.parent.name}.png",

                "score": float(scores[i, gid - 1]), "margin": float(margins[i]),

                "needs_review": bool(margins[i] < 0.02)}

               for i, (p, gid) in enumerate(zip(originals, groups))]

    categories = [{"id": i + 1, "name": name, "count": int((groups == i + 1).sum()),

                   "prompts": list(descriptions)} for i, (name, descriptions) in enumerate(CATEGORIES)]

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".semantic-", dir=output.parent) as temporary:

        stage = Path(temporary)

        for item in categories:

            (stage / str(item["id"])).mkdir()

        for path, row in zip(originals, records):

            shutil.copy2(path, stage / row["file"])

        np.savez_compressed(stage / "features.npz", ids=np.asarray([p.parent.name for p in originals]),

                            source_sha256=np.asarray(hashes), embeddings=features, class_embeddings=classes,

                            similarities=scores, groups=groups, margins=margins)

        manifest = {"model": MODEL, "method": "CLIP zero-shot semantic classification (6 categories)",

                    "input": str(source), "total": len(originals), "n_groups": len(CATEGORIES),

                    "categories": categories, "images": records,

                    "note": "score 为余弦相似度；margin 为前两类分差。needs_review 是低分差复核提示，不是校准概率。"}

        (stage / "group_semantic.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        backup = None

        if output.exists():

            backup = output.with_name(output.name + ".previous-" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

            output.rename(backup)

        try:

            stage.rename(output)

        except OSError:

            if backup and not output.exists():

                backup.rename(output)

            raise

    print(f"完成：{len(originals)} 张图片，{len(categories)} 个类别。输出：{output}")

    if backup:

        print(f"旧结果备份：{backup}")

    for item in categories:

        print(f"{item['id']} {item['name']}: {item['count']} 张")





if __name__ == "__main__":

    main()
