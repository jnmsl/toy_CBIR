import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def _pid(path):
    """42431.jpg -> '42431'"""
    return os.path.splitext(os.path.basename(path))[0]


def run_evaluation(cbir, num_queries=200, top_k=10, results_dir="./results"):
    """Precision@K by category level using styles.csv metadata."""
    if not cbir.catalog:
        print("No catalog loaded, skipping evaluation.")
        return

    candidates = [p for p in cbir.image_paths if _pid(p) in cbir.catalog]
    queries = random.sample(candidates, min(num_queries, len(candidates)))

    levels = ["masterCategory", "subCategory", "articleType", "baseColour"]
    precision = {lv: {k: [] for k in [1, 5, 10]} for lv in levels}

    print(f"\nEvaluating {len(queries)} queries (top-{top_k})...")
    for qpath in tqdm(queries, desc="Eval"):
        qmeta = cbir.catalog[_pid(qpath)]
        results = cbir.search(qpath, top_k=top_k)

        for lv in levels:
            q_val = qmeta.get(lv, "").strip()
            if not q_val:
                continue
            for k in [1, 5, 10]:
                hits = sum(
                    1 for rp, _ in results[:k]
                    if cbir.catalog.get(_pid(rp), {}).get(lv, "").strip() == q_val
                )
                precision[lv][k].append(hits / k)

    # print report
    os.makedirs(results_dir, exist_ok=True)
    lines = [
        "=" * 60,
        "  Fashion CBIR - Evaluation Report",
        f"  Queries: {len(queries)}  |  Indexed: {len(cbir.image_paths)}",
        "=" * 60,
    ]
    for lv in levels:
        lines.append(f"\n  {lv}:")
        for k in [1, 5, 10]:
            vals = precision[lv][k]
            avg = np.mean(vals) if vals else 0.0
            lines.append(f"    Precision@{k:2d} = {avg:.4f}  (n={len(vals)})")

    report = "\n".join(lines)
    print(report)

    path = os.path.join(results_dir, "evaluation_report.txt")
    with open(path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved: {path}")

    _plot_precision(precision, levels, results_dir)


def _plot_precision(precision, levels, results_dir):
    ks = [1, 5, 10]
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 4))
    colours = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, lv in enumerate(levels):
        vals = [np.mean(precision[lv][k]) if precision[lv][k] else 0 for k in ks]
        axes[i].bar([f"@{k}" for k in ks], vals, color=colours)
        axes[i].set_title(lv, fontsize=11)
        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel("Precision")
        for j, v in enumerate(vals):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    plt.suptitle("Precision@K by Category Level", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(results_dir, "precision_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def visualize_query(cbir, query_path, results, save_path):
    """Save query + top results as image grid."""
    n = min(len(results), 5)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.5 * (n + 1), 4.5))

    def show(ax, path, title, color="black"):
        im = cv2.imread(path)
        if im is not None:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=8, color=color)
        ax.axis("off")

    qm = cbir.catalog.get(_pid(query_path), {})
    show(axes[0], query_path,
         f"QUERY\n{qm.get('articleType','?')}\n{qm.get('baseColour','?')}",
         color="red")

    for i in range(n):
        rp, d = results[i]
        rm = cbir.catalog.get(_pid(rp), {})
        show(axes[i + 1], rp,
             f"#{i+1}  d={d:.2f}\n{rm.get('articleType','?')}\n{rm.get('baseColour','?')}")

    plt.suptitle(f"Query: {os.path.basename(query_path)}", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_demo_queries(cbir, num_per_cat=2, top_k=5, results_dir="./results"):
    if not cbir.catalog:
        print("No catalog, skipping demo.")
        return

    os.makedirs(results_dir, exist_ok=True)

    by_type = {}
    for p in cbir.image_paths:
        atype = cbir.catalog.get(_pid(p), {}).get("articleType", "").strip()
        if atype:
            by_type.setdefault(atype, []).append(p)

    top_cats = sorted(by_type, key=lambda t: len(by_type[t]), reverse=True)[:8]
    print(f"\nDemo queries from {len(top_cats)} categories...")

    for cat in top_cats:
        samples = random.sample(by_type[cat], min(num_per_cat, len(by_type[cat])))
        for j, qpath in enumerate(samples):
            results = cbir.search(qpath, top_k=top_k)
            safe = cat.replace(" ", "_").replace("/", "_")
            fname = f"demo_{safe}_{j+1}.png"
            save_path = os.path.join(results_dir, fname)
            visualize_query(cbir, qpath, results, save_path)
            print(f"  {fname}")
