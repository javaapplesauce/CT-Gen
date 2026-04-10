#!/usr/bin/env python3
"""
CT-Gen FSDP — Download CT-RATE data + preprocess NIfTIs to float16 .pt tensors.

STREAMING PIPELINE: Downloads volumes in small chunks, preprocesses them in
parallel, and deletes raw NIfTIs immediately. Peak raw disk usage is ~50 GB
(one chunk) instead of 21 TB (entire dataset).

Run once (single-process, before training):
    export HF_TOKEN=hf_...
    python download_and_preprocess.py

Override config from CLI:
    python download_and_preprocess.py data.volume_limit=500 data.chunk_size=20

Storage (full dataset, ~21,000 volumes):
    Raw NIfTI (temporary, per chunk): ~50 GB peak
    Preprocessed .pt (final):         ~370 GB total
"""
import os
import sys
import glob
import shutil
import subprocess
import multiprocessing as mp
from functools import partial

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged, Resized,
)

# ── hf_transfer must be enabled before importing huggingface_hub ──
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download


# MONAI preprocessing pipeline
CT_TRANSFORMS = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400,
                         b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=["image"], spatial_size=(480, 480, 40)),
])


def vol_name_to_hf_path(name: str) -> str:
    """Convert volume name to HuggingFace repo path.

    CT-RATE uses two splits:
      train_*  → dataset/train/{patient}/{series}/{name}.nii.gz
      valid_*  → dataset/valid_fixed/{patient}/{series}/{name}.nii.gz
    """
    name    = name.replace(".nii.gz", "")
    parts   = name.split("_")          # ["train","1","a","1"] or ["valid","1","a","1"]
    patient = "_".join(parts[:2])      # "train_1" / "valid_1"
    series  = "_".join(parts[:3])      # "train_1_a" / "valid_1_a"
    if parts[0] == "train":
        return f"dataset/train/{patient}/{series}/{name}.nii.gz"
    return f"dataset/valid_fixed/{patient}/{series}/{name}.nii.gz"


# Worker function for parallel preprocessing
def _preprocess_one(args):
    """Preprocess a single NIfTI → float16 .pt. Runs in a worker process."""
    vn, raw_path, pt_path = args
    try:
        data   = CT_TRANSFORMS({"image": raw_path})["image"]  # (C, H, W, S) MetaTensor
        # .as_tensor() strips the MONAI MetaTensor wrapper → plain torch.Tensor.
        # Without this, torch.load(..., weights_only=True) refuses to unpickle the file
        # in PyTorch >=2.6 because MetaTensor is not an allowlisted global.
        tensor = data.as_tensor().permute(0, 3, 1, 2).half()  # (C, S, H, W) fp16
        torch.save(tensor, pt_path)
        return (vn, None)
    except Exception as e:
        return (vn, str(e))


def _download_chunk(chunk, base_url, hf_token):
    """Download a chunk of volumes via aria2c (fire-and-forget; caller checks disk)."""
    url_file = "/tmp/aria2_urls.txt"
    all_dirs = set()
    entries  = []

    for vn, hf_path, raw_path, pt_path in chunk:
        all_dirs.add(os.path.dirname(raw_path))
        entries.append((f"{base_url}/{hf_path}", raw_path))

    for d in all_dirs:
        os.makedirs(d, exist_ok=True)

    # aria2c's `out=` must be a filename only — directory components are
    # set via `dir=`.  Writing a full absolute path to `out=` causes aria2c
    # to treat the whole string as a relative filename and save to CWD.
    with open(url_file, "w") as f:
        for url, dest in entries:
            f.write(
                f"{url}\n"
                f"  dir={os.path.dirname(dest)}\n"
                f"  out={os.path.basename(dest)}\n"
                f"  header=Authorization: Bearer {hf_token}\n"
            )

    cmd = [
        "aria2c",
        "--input-file",                url_file,
        "--max-concurrent-downloads",  "64",
        "--split",                     "16",
        "--min-split-size",            "1M",
        "--max-connection-per-server", "16",
        "--file-allocation",           "none",
        "--disk-cache",                "512M",
        "--no-conf",
        "--allow-overwrite",           "true",
        "--auto-file-renaming",        "false",
        "--max-tries",                 "5",
        "--retry-wait",                "3",
        "--connect-timeout",           "30",
        "--timeout",                   "600",
        "--summary-interval",          "15",
        "--console-log-level",         "warn",
    ]
    subprocess.run(cmd, check=True)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    raw_dir         = cfg.paths.raw_dir
    pt_dir          = cfg.paths.pt_dir
    models_dir      = os.path.dirname(cfg.paths.ct_clip_weights)
    ct_clip_weights = cfg.paths.ct_clip_weights
    repo_id          = cfg.data.repo_id
    reports_hf_paths = [cfg.data.reports_hf_path]
    # Also pull training reports when the key exists (full CT-RATE dataset)
    train_hf = getattr(cfg.data, "train_reports_hf_path", None)
    if train_hf:
        reports_hf_paths.append(train_hf)
    volume_limit    = cfg.data.volume_limit
    chunk_size      = cfg.data.chunk_size
    prep_workers    = cfg.data.preprocess_workers

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: Set HF_TOKEN environment variable.")
        print("  export HF_TOKEN=hf_...")
        sys.exit(1)

    for d in [raw_dir, pt_dir, models_dir]:
        os.makedirs(d, exist_ok=True)

    num_workers = prep_workers or min(mp.cpu_count(), 32)
    print(f"Preprocessing workers: {num_workers}")

    # CT-CLIP weights
    if not os.path.exists(ct_clip_weights):
        print("Downloading CT-CLIP v2 weights...")
        hf_hub_download(
            repo_id="ibrahimhamamci/CT-RATE",
            filename="models/CT-CLIP-Related/CT-CLIP_v2.pt",
            repo_type="dataset", token=hf_token,
            local_dir=models_dir,
        )
        found = glob.glob(os.path.join(models_dir, "**", "*CT-CLIP_v2*.pt"), recursive=True)
        if found and found[0] != ct_clip_weights:
            shutil.copy(found[0], ct_clip_weights)
        print(f"  Weights: {ct_clip_weights}")
    else:
        print(f"CT-CLIP weights present: {ct_clip_weights}")

    # Reports CSVs (validation + optional training split)
    frames = []
    for hf_path in reports_hf_paths:
        local_csv = os.path.join(raw_dir, hf_path)
        if not os.path.exists(local_csv):
            print(f"Downloading reports CSV: {hf_path} ...")
            hf_hub_download(
                repo_id=repo_id, filename=hf_path,
                repo_type="dataset", token=hf_token, local_dir=raw_dir,
            )
        print(f"  Reports CSV: {local_csv}")
        frames.append(pd.read_csv(local_csv))

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    vol_col   = df.columns[0]
    vol_names = (
        df[vol_col].astype(str)
        .str.replace(".nii.gz", "", regex=False)
        .drop_duplicates().tolist()
    )
    if volume_limit:
        vol_names = vol_names[:volume_limit]
    print(f"  Total volumes in dataset: {len(vol_names)}")

    # Build work list (skip already-processed)
    work = []
    for vn in vol_names:
        pt_path = os.path.join(pt_dir, vn + ".pt")
        if os.path.exists(pt_path) and os.path.getsize(pt_path) > 0:
            continue
        hf_path  = vol_name_to_hf_path(vn)
        raw_path = os.path.join(raw_dir, hf_path)
        work.append((vn, hf_path, raw_path, pt_path))

    already_done = len(vol_names) - len(work)
    if already_done:
        print(f"  Already preprocessed: {already_done}")

    if not work:
        print("All volumes already preprocessed.")
        _print_summary(pt_dir, raw_dir)
        return

    # Streaming chunked pipeline
    base_url   = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    num_chunks = (len(work) + chunk_size - 1) // chunk_size
    total_ok   = 0
    total_fail = 0

    print(f"\nStreaming pipeline: {len(work)} volumes in {num_chunks} chunks "
          f"of {chunk_size}")
    print(f"  Peak raw disk per chunk: ~{chunk_size} GB")
    print(f"  Raw NIfTIs auto-deleted after each chunk\n")

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, len(work))
        chunk = work[start:end]

        print(f"── Chunk {chunk_idx + 1}/{num_chunks} "
              f"({len(chunk)} volumes) ──")

        # Download
        # Separate volumes that already have raw files on disk (e.g. from
        # a previous interrupted run) from those that need downloading.
        to_download = []
        already_raw = []
        for item in chunk:
            vn, hf_path, raw_path, pt_path = item
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                already_raw.append(item)
            else:
                to_download.append(item)

        if to_download:
            print(f"  Downloading {len(to_download)} volumes...")
            _download_chunk(to_download, base_url, hf_token)
        if already_raw:
            print(f"  {len(already_raw)} volumes already downloaded (raw on disk)")

        # Preprocess in parallel
        # Re-check which raw files actually exist after download
        preprocess_args = []
        for vn, hf_path, raw_path, pt_path in chunk:
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                preprocess_args.append((vn, raw_path, pt_path))

        if preprocess_args:
            print(f"  Preprocessing {len(preprocess_args)} volumes "
                  f"({num_workers} workers)...")
            with mp.Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_preprocess_one, preprocess_args),
                    total=len(preprocess_args),
                    desc=f"  Chunk {chunk_idx + 1}",
                ))

            ok   = sum(1 for _, err in results if err is None)
            fail = [(vn, err) for vn, err in results if err is not None]
            total_ok   += ok
            total_fail += len(fail)
            print(f"  Preprocessed: {ok} ok, {len(fail)} failed")
            if fail:
                for vn, err in fail[:5]:
                    print(f"    FAIL: {vn}: {err}")

        # Delete raw NIfTIs for this chunk
        deleted = 0
        for vn, hf_path, raw_path, pt_path in chunk:
            if os.path.exists(raw_path):
                os.remove(raw_path)
                deleted += 1
            # Clean up empty parent directories
            parent = os.path.dirname(raw_path)
            for _ in range(3):  # up to 3 levels
                try:
                    os.rmdir(parent)  # only removes if empty
                    parent = os.path.dirname(parent)
                except OSError:
                    break
        if deleted:
            print(f"  Deleted {deleted} raw NIfTI files")
        print()

    print(f"Pipeline complete: {total_ok} preprocessed, {total_fail} failed")
    _print_summary(pt_dir, raw_dir)


def _print_summary(pt_dir, raw_dir):
    pt_files = glob.glob(os.path.join(pt_dir, "*.pt"))
    pt_bytes = sum(os.path.getsize(p) for p in pt_files)
    raw_files = glob.glob(os.path.join(raw_dir, "**", "*.nii.gz"), recursive=True)
    raw_bytes = sum(os.path.getsize(p) for p in raw_files)
    print(f"\nPreprocessed: {len(pt_files)} tensors  |  {pt_bytes / 1e9:.1f} GB")
    if raw_bytes:
        print(f"Residual raw NIfTIs: {len(raw_files)} files  |  "
              f"{raw_bytes / 1e9:.1f} GB")
    else:
        print("No raw NIfTIs on disk (all cleaned up)")


if __name__ == "__main__":
    main()
