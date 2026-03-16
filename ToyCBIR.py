import os
import sys
import csv
import random
import pickle
import argparse
import time

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

from features import (IMG_SIZE, color_histogram, lbp_descriptor,
                      cnn_features_batch, extract_features, get_model)
from evaluate import run_evaluation, run_demo_queries, visualize_query


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FashionCBIR:

    def __init__(self, index_dir="index", strategy="cascaded"):
        self.index_dir = index_dir
        self.strategy = strategy
        self.image_paths = []
        self.catalog = {}

        # separate feature matrices for cascaded retrieval
        self.semantic_matrix = None
        self.color_matrix = None
        self.texture_matrix = None

        # kNN indexes
        self.semantic_nn = None
        self.color_nn = None

        get_model()

    def _build_nn(self):
        """Build nearest-neighbor indexes for semantic and color features."""
        self.semantic_nn = NearestNeighbors(
            algorithm="brute", metric="cosine", n_jobs=-1)
        self.semantic_nn.fit(self.semantic_matrix)

        self.color_nn = NearestNeighbors(
            algorithm="brute", metric="euclidean", n_jobs=-1)
        self.color_nn.fit(self.color_matrix)

    def load_catalog(self, csv_path):
        self.catalog = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row.get("id", "").strip()
                if pid:
                    self.catalog[pid] = row
        print(f"Catalog: {len(self.catalog)} products")

    @staticmethod
    def pid(path):
        return os.path.splitext(os.path.basename(path))[0]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_folder(self, folder, batch_size=64, max_images=None):
        print(f"Scanning {folder}...")
        files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXTS
        )
        if max_images:
            files = files[:max_images]
        print(f"Found {len(files)} images.")

        all_cnn, all_color, all_lbp = [], [], []
        valid_paths = []

        t0 = time.time()
        for start in tqdm(range(0, len(files), batch_size), desc="Indexing"):
            batch_files = files[start:start + batch_size]
            imgs, paths = [], []

            for f in batch_files:
                p = os.path.join(folder, f)
                im = cv2.imread(p)
                if im is None:
                    continue
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
                imgs.append(rgb)
                paths.append(p)

            if not imgs:
                continue

            # CNN on GPU as a batch
            cnn = cnn_features_batch(imgs)
            all_cnn.append(cnn)

            # classical features per image
            for im in imgs:
                gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
                all_color.append(color_histogram(hsv))
                all_lbp.append(lbp_descriptor(gray))

            valid_paths.extend(paths)

        if not valid_paths:
            print("No valid images found!")
            return

        self.semantic_matrix = np.vstack(all_cnn).astype("float32")
        self.color_matrix = np.array(all_color, dtype="float32")
        self.texture_matrix = np.array(all_lbp, dtype="float32")
        self.image_paths = valid_paths

        elapsed = time.time() - t0
        print(f"Indexed {len(valid_paths)} images ({elapsed:.0f}s)")
        print(f"  semantic {self.semantic_matrix.shape}  "
              f"color {self.color_matrix.shape}  "
              f"texture {self.texture_matrix.shape}")

        self._build_nn()
        self._save_index()

    def _save_index(self):
        os.makedirs(self.index_dir, exist_ok=True)
        np.save(os.path.join(self.index_dir, "semantic.npy"), self.semantic_matrix)
        np.save(os.path.join(self.index_dir, "color.npy"), self.color_matrix)
        np.save(os.path.join(self.index_dir, "texture.npy"), self.texture_matrix)
        with open(os.path.join(self.index_dir, "paths.pkl"), "wb") as f:
            pickle.dump(self.image_paths, f)
        print(f"Saved index to {self.index_dir}/")

    def load_index(self):
        sem_path = os.path.join(self.index_dir, "semantic.npy")
        pkl_path = os.path.join(self.index_dir, "paths.pkl")
        if not os.path.exists(sem_path) or not os.path.exists(pkl_path):
            return False
        print("Loading index...")
        self.semantic_matrix = np.load(sem_path)
        self.color_matrix = np.load(os.path.join(self.index_dir, "color.npy"))
        self.texture_matrix = np.load(os.path.join(self.index_dir, "texture.npy"))
        with open(pkl_path, "rb") as f:
            self.image_paths = pickle.load(f)
        self._build_nn()
        print(f"Loaded {len(self.image_paths)} images.")
        return True

    # ------------------------------------------------------------------
    # Search strategies
    # ------------------------------------------------------------------

    def search(self, query_path, top_k=10):
        feats = extract_features(query_path)
        if feats is None:
            return []

        if self.strategy == "resnet_only":
            return self._search_semantic(feats, query_path, top_k)
        elif self.strategy == "color_only":
            return self._search_color(feats, query_path, top_k)
        else:
            return self._search_cascaded(feats, query_path, top_k)

    def _search_semantic(self, feats, query_path, top_k):
        """ResNet-only retrieval using cosine distance."""
        k = min(top_k + 1, len(self.image_paths))
        dists, ids = self.semantic_nn.kneighbors(
            feats["semantic"].reshape(1, -1), n_neighbors=k)
        return [
            (self.image_paths[i], dists[0][j])
            for j, i in enumerate(ids[0])
            if self.image_paths[i] != query_path
        ][:top_k]

    def _search_color(self, feats, query_path, top_k):
        """Color-only retrieval using Euclidean distance on HSV histograms."""
        k = min(top_k + 1, len(self.image_paths))
        dists, ids = self.color_nn.kneighbors(
            feats["color"].reshape(1, -1), n_neighbors=k)
        return [
            (self.image_paths[i], dists[0][j])
            for j, i in enumerate(ids[0])
            if self.image_paths[i] != query_path
        ][:top_k]

    def _search_cascaded(self, feats, query_path, top_k):
        """Two-stage: semantic filter (top 100) then color+texture re-rank."""
        # stage 1: cosine similarity on CNN embeddings
        n_cand = min(101, len(self.image_paths))
        _, ids = self.semantic_nn.kneighbors(
            feats["semantic"].reshape(1, -1), n_neighbors=n_cand)

        # stage 2: re-rank candidates by color + texture distance
        scored = []
        for idx in ids[0]:
            if self.image_paths[idx] == query_path:
                continue
            d_color = distance.euclidean(feats["color"], self.color_matrix[idx])
            d_texture = distance.euclidean(feats["texture"], self.texture_matrix[idx])
            scored.append((self.image_paths[idx], d_color + 0.5 * d_texture))

        scored.sort(key=lambda x: x[1])
        return scored[:top_k]


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fashion CBIR")
    parser.add_argument("--mode",
                        choices=["index", "search", "evaluate", "demo", "all"],
                        default="all")
    parser.add_argument("--strategy",
                        choices=["cascaded", "resnet_only", "color_only"],
                        default="cascaded")
    parser.add_argument("--image-dir", default="./images")
    parser.add_argument("--metadata", default="./archive/styles.csv")
    parser.add_argument("--index-dir", default="./index")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-eval", type=int, default=200)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    os.makedirs(args.results_dir, exist_ok=True)

    cbir = FashionCBIR(index_dir=args.index_dir, strategy=args.strategy)

    if os.path.exists(args.metadata):
        cbir.load_catalog(args.metadata)
    else:
        print(f"Warning: metadata not found at {args.metadata}")

    # index or load
    if args.mode in ("index", "all"):
        if not cbir.load_index():
            cbir.index_folder(args.image_dir, batch_size=args.batch_size,
                              max_images=args.max_images)
    else:
        if not cbir.load_index():
            print("No index found. Run --mode index first.")
            sys.exit(1)

    # single query
    if args.mode == "search" and args.query:
        results = cbir.search(args.query, top_k=args.top_k)
        print(f"\nResults for {args.query} (strategy={args.strategy}):")
        for i, (p, d) in enumerate(results):
            m = cbir.catalog.get(cbir.pid(p), {})
            print(f"  {i+1}. {os.path.basename(p)}  d={d:.4f}  "
                  f"{m.get('articleType','')} / {m.get('baseColour','')}")
        save = os.path.join(args.results_dir, "search_result.png")
        visualize_query(cbir, args.query, results[:5], save)

    # evaluation
    if args.mode in ("evaluate", "all"):
        run_evaluation(cbir, num_queries=args.num_eval,
                       top_k=args.top_k, results_dir=args.results_dir)

    # demo
    if args.mode in ("demo", "all"):
        run_demo_queries(cbir, top_k=5, results_dir=args.results_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
