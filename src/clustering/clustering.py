"""
CodeBERT Embedding + KMeans 聚类

用途：
- 从 JSONL 文件（如 data/gpt_out_algorithm.jsonl）中抽取算法描述文本
- 使用 CodeBERT 生成句向量
- 执行 KMeans 聚类，并输出结果 CSV

运行示例：
    python -m src.clustering.clustering \
        --jsonl data/gpt_out_algorithm.jsonl \
        --n-clusters 8 \
        --out data/algorithm_clusters.csv
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer


# ---------------------------
# 加载模型
# ---------------------------
def load_codebert():
    print("Loading CodeBERT model...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.eval()
    return tokenizer, model


# ---------------------------
# 数据加载与解析
# ---------------------------
def _gather_text_from_output(output_obj: dict) -> str:
    """从 OpenAI Responses API 风格的 output 对象提取纯文本。"""
    if not isinstance(output_obj, dict):
        return ""
    # 典型结构：{"type": "message", "content": [{"type": "output_text", "text": "..."}, ...]}
    parts = []
    content = output_obj.get("content") or []
    for c in content:
        t = c.get("text") if isinstance(c, dict) else None
        if isinstance(t, str) and t.strip():
            parts.append(t)
    return "\n".join(parts).strip()


def load_algorithms_from_jsonl(path: str, limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    读取 JSONL，每行一条对象，返回：
    - texts: 算法描述文本列表
    - ids:   用于标识的 ID（优先 custom_id，其次 id，最后为序号）
    """
    texts: List[str] = []
    ids: List[str] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 跳过非 JSON 行
                continue

            rid = obj.get("custom_id") or obj.get("id") or str(idx)
            body = (obj.get("response") or {}).get("body") or {}
            outputs = body.get("output") or []

            merged = []
            for out in outputs:
                txt = _gather_text_from_output(out)
                if txt:
                    merged.append(txt)
            full_text = "\n".join(merged).strip()
            if full_text:
                ids.append(str(rid))
                texts.append(full_text)
                if limit is not None and len(texts) >= limit:
                    break

    if not texts:
        raise ValueError("未从 JSONL 中解析到任何算法文本。请检查数据格式。")

    return texts, ids


# ---------------------------
# 文本预处理（可选的小清洗/标题抽取）
# ---------------------------
def extract_title(text: str) -> str:
    """尽量从文本中抽出算法标题/名称，用于结果展示。"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""

    # 常见格式：第一段是 "1. **Algorithm Name**"，下一行是真正的名称
    for i, line in enumerate(lines[:-1]):
        if re.search(r"algorithm\s*name", line, flags=re.I):
            name_line = lines[i + 1]
            # 去掉粗体等 markdown 修饰
            name_line = re.sub(r"[*`_#>]+", "", name_line).strip()
            if name_line:
                return name_line

    # 回退：找第一行不是目录/编号的内容
    for line in lines:
        if not re.match(r"^\d+\.|^[-*+]\s|^#+\s", line):
            return re.sub(r"[*`_#>]+", "", line).strip()

    return re.sub(r"[*`_#>]+", "", lines[0]).strip()


# ---------------------------
# Embedding 生成（批处理 + 平均池化）
# ---------------------------
@torch.no_grad()
def encode_texts(tokenizer, model, texts: List[str], batch_size: int = 8) -> np.ndarray:
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        outputs = model(**inputs)
        # 平均池化到句级
        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_vecs.append(emb)
    X = np.vstack(all_vecs)
    X = normalize(X)
    return X


# ---------------------------
# 评估与 K 选择
# ---------------------------
def sweep_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    silhouette_sample: Optional[int] = None,
    random_state: int = 42,
):
    n = X.shape[0]
    k_min = max(2, k_min)
    k_max = min(max(k_min, k_max), n - 1)
    rows = []
    print(f"\nEvaluating K in [{k_min}, {k_max}] on {n} samples...")
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        inertia = float(kmeans.inertia_)
        try:
            sil = float(
                silhouette_score(
                    X, labels, metric="euclidean", sample_size=silhouette_sample, random_state=random_state
                )
            )
        except Exception:
            sil = float("nan")
        try:
            ch = float(calinski_harabasz_score(X, labels))
        except Exception:
            ch = float("nan")
        try:
            db = float(davies_bouldin_score(X, labels))
        except Exception:
            db = float("nan")
        rows.append({"k": k, "silhouette": sil, "calinski": ch, "davies": db, "inertia": inertia})

    # 推荐：优先最大 silhouette；若无效，用最小 DB；再用最大 CH
    best_k = None
    valid_sil = [r for r in rows if not np.isnan(r["silhouette"])]
    if valid_sil:
        best_k = max(valid_sil, key=lambda r: r["silhouette"]) ["k"]
    else:
        valid_db = [r for r in rows if not np.isnan(r["davies"])]
        if valid_db:
            best_k = min(valid_db, key=lambda r: r["davies"]) ["k"]
        else:
            valid_ch = [r for r in rows if not np.isnan(r["calinski"])]
            if valid_ch:
                best_k = max(valid_ch, key=lambda r: r["calinski"]) ["k"]

    # 打印表格
    print("\nK sweep (higher silhouette/Calinski better; lower Davies/inertia better):")
    header = f"{'K':>3}  {'silhouette':>10}  {'calinski':>12}  {'davies':>8}  {'inertia':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['k']:>3}  {r['silhouette']:>10.4f}  {r['calinski']:>12.2f}  {r['davies']:>8.4f}  {r['inertia']:>12.2f}"
        )

    if best_k is not None:
        print(f"\nRecommended K ≈ {best_k} (by metrics above)")
    else:
        print("\nNo clear recommendation (metrics unavailable).")

    return best_k, rows


# ---------------------------
# 主流程
# ---------------------------
def run(jsonl: str, n_clusters: int, out_csv: Optional[str] = None, limit: Optional[int] = None):
    texts, ids = load_algorithms_from_jsonl(jsonl, limit=limit)
    print(f"Loaded {len(texts)} algorithms from {jsonl}")

    tokenizer, model = load_codebert()
    X = encode_texts(tokenizer, model, texts, batch_size=8)

    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # 汇总
    print("\n聚类结果：")
    titles = [extract_title(t) for t in texts]
    for rid, lab, title in zip(ids, labels, titles):
        title_disp = title if title else (texts[0][:60] + "...")
        print(f"[Cluster {lab}] {rid} :: {title_disp}")

    print("\nCluster 分布：")
    for c in range(n_clusters):
        count = int(np.sum(labels == c))
        print(f"Cluster {c}: {count} 个样本")

    if out_csv:
        import csv

        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "cluster", "title"])  # 简要输出
            for rid, lab, title in zip(ids, labels, titles):
                w.writerow([rid, lab, title])
        print(f"\nSaved clustering to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster algorithms from JSONL using CodeBERT + KMeans")
    parser.add_argument("--jsonl", type=str, default="data/gpt_out_algorithm.jsonl", help="输入 JSONL 文件路径")
    parser.add_argument("--n-clusters", type=int, default=8, help="KMeans 聚类数")
    parser.add_argument("--out", type=str, default="data/algorithm_clusters.csv", help="结果 CSV 输出路径")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条（调试/快速验证）")
    parser.add_argument("--sweep-k", action="store_true", help="扫描一组 K 并输出指标与推荐值")
    parser.add_argument("--k-min", type=int, default=2, help="扫描 K 的下界（含）")
    parser.add_argument("--k-max", type=int, default=15, help="扫描 K 的上界（含）")
    parser.add_argument("--silhouette-sample", type=int, default=None, help="silhouette 计算的样本数上限")
    args = parser.parse_args()

    if args.sweep_k:
        texts, _ = load_algorithms_from_jsonl(args.jsonl, limit=args.limit)
        tokenizer, model = load_codebert()
        X = encode_texts(tokenizer, model, texts, batch_size=8)
        sweep_k(
            X,
            k_min=args.k_min,
            k_max=args.k_max,
            silhouette_sample=args.silhouette_sample,
            random_state=42,
        )
    else:
        run(args.jsonl, args.n_clusters, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
