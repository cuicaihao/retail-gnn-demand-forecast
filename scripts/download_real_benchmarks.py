#!/usr/bin/env python3
"""
Download and Cache Real-World Public Benchmark Datasets:
1. Dingdong FreshRetailNet-50K (HuggingFace: Dingdong-Inc/FreshRetailNet-50K)
2. Kaggle Corporación Favorita (HuggingFace: khadidia-77/favorita)
3. Walmart M5 Forecasting (HuggingFace: denephew/M5_Forecasting)
"""

import os
import sys
import urllib.request
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(REPO_ROOT, "data", "public_benchmarks", "raw")


def download_file_with_progress(url: str, dest_path: str, label: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"[{label}] Already exists: {dest_path} ({size_mb:.2f} MB), skipping download.")
        return

    print(f"[{label}] Downloading from {url} -> {dest_path}...")
    t0 = time.time()

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = (count * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r[{label}] Progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook)
    elapsed = time.time() - t0
    final_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"\n[{label}] Done: {final_mb:.2f} MB in {elapsed:.1f}s ({final_mb / max(0.1, elapsed):.2f} MB/s)")


def download_freshretailnet():
    base_url = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/data"
    dest_dir = os.path.join(RAW_DATA_DIR, "freshretailnet")
    download_file_with_progress(
        f"{base_url}/eval.parquet",
        os.path.join(dest_dir, "eval.parquet"),
        "FreshRetailNet-50K Eval"
    )
    download_file_with_progress(
        f"{base_url}/train.parquet",
        os.path.join(dest_dir, "train.parquet"),
        "FreshRetailNet-50K Train"
    )


def download_favorita():
    base_url_raw = "https://huggingface.co/datasets/khadidia-77/favorita/raw/main"
    base_url_resolve = "https://huggingface.co/datasets/khadidia-77/favorita/resolve/main"
    dest_dir = os.path.join(RAW_DATA_DIR, "favorita")
    for f in ["stores.csv", "oil.csv", "holidays_events.csv", "transactions.csv"]:
        download_file_with_progress(
            f"{base_url_raw}/{f}",
            os.path.join(dest_dir, f),
            f"Favorita {f}"
        )
    download_file_with_progress(
        f"{base_url_resolve}/train_last10w.parquet",
        os.path.join(dest_dir, "train_last10w.parquet"),
        "Favorita train_last10w.parquet"
    )


def download_walmart_m5():
    base_url = "https://huggingface.co/datasets/denephew/M5_Forecasting/resolve/main"
    dest_dir = os.path.join(RAW_DATA_DIR, "walmart_m5")
    download_file_with_progress(
        f"{base_url}/calendar.csv",
        os.path.join(dest_dir, "calendar.csv"),
        "Walmart M5 Calendar"
    )
    download_file_with_progress(
        f"{base_url}/sales_train_validation.csv",
        os.path.join(dest_dir, "sales_train_validation.csv"),
        "Walmart M5 Sales"
    )


def main():
    print("=" * 70)
    print("Downloading Real Public Datasets for Retail GNN Benchmark")
    print("=" * 70)
    download_freshretailnet()
    download_favorita()
    download_walmart_m5()
    print("=" * 70)
    print("All datasets downloaded successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
