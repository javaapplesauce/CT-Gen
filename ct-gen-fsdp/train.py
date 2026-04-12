#!/usr/bin/env python3
"""
CT-Gen FSDP — Multi-GPU training script.

Launch with:
    accelerate launch --config_file $CTGEN_BASE/accel/fsdp_config.yaml train.py

Override config from CLI:
    accelerate launch ... train.py stage1.lr=3e-4 model.llm_id=meta-llama/Meta-Llama-3-8B-Instruct
"""
import os
import sys
import json
import glob
import functools
from contextlib import nullcontext

import nltk
import hydra
import pandas as pd
import torch
import torch.nn as nn
import wandb
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, FullyShardedDataParallel as FSDP_class
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def load_config(overrides=None):
    """Load conf/config.yaml and resolve interpolations.

    If a train_config.yaml was saved in the checkpoint directory (written by
    train.py's main()), it is loaded instead so that evaluate.py and
    inference.py automatically use the same architecture / hyperparameters as
    the training run — including any CLI overrides like llm_hidden_dim.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_cfg = OmegaConf.load(os.path.join(script_dir, "conf", "config.yaml"))
    OmegaConf.resolve(base_cfg)

    saved = os.path.join(base_cfg.paths.ckpt_dir, "train_config.yaml")
    if os.path.exists(saved):
        cfg = OmegaConf.load(saved)
        print(f"[load_config] Loaded saved config from {saved}")
        print(f"[load_config]   llm_id={cfg.model.llm_id}  llm_hidden_dim={cfg.model.llm_hidden_dim}")
    else:
        cfg = base_cfg
        print(f"[load_config] No saved config found at {saved}, using base config")
        print(f"[load_config]   llm_id={cfg.model.llm_id}  llm_hidden_dim={cfg.model.llm_hidden_dim}")
        print(f"[load_config]   If this is wrong, run the backfill cell or a training run first.")

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def compute_metrics(generated: list, references: list) -> dict:
    """Compute BLEU, ROUGE, METEOR, BERTScore over (generated, reference) pairs."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score_fn

    for resource in ["wordnet", "omw-1.4", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{resource}" if "punkt" not in resource
                           else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    smoothing = SmoothingFunction().method1
    scorer    = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    bleu1_scores  = []
    bleu4_scores  = []
    meteor_scores = []
    rouge_scores  = defaultdict(list)

    for gen, ref in zip(generated, references):
        gen_tokens = gen.lower().split()
        ref_tokens = ref.lower().split()
        if not gen_tokens or not ref_tokens:
            continue
        bleu1_scores.append(sentence_bleu(
            [ref_tokens], gen_tokens, weights=(1, 0, 0, 0),
            smoothing_function=smoothing,
        ))
        bleu4_scores.append(sentence_bleu(
            [ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        ))
        meteor_scores.append(nltk_meteor([ref_tokens], gen_tokens))
        r = scorer.score(ref, gen)
        for key in ["rouge1", "rouge2", "rougeL"]:
            rouge_scores[key].append(r[key].fmeasure)

    P, R, F1 = bert_score_fn(generated, references, lang="en", verbose=False, batch_size=16)

    n = len(bleu1_scores)
    return {
        "n_samples":    len(generated),
        "n_scored":     n,
        "bleu_1":       sum(bleu1_scores)  / max(n, 1),
        "bleu_4":       sum(bleu4_scores)  / max(n, 1),
        "meteor":       sum(meteor_scores) / max(n, 1),
        "rouge_1":      sum(rouge_scores["rouge1"]) / max(n, 1),
        "rouge_2":      sum(rouge_scores["rouge2"]) / max(n, 1),
        "rouge_L":      sum(rouge_scores["rougeL"]) / max(n, 1),
        "bertscore_p":  P.mean().item(),
        "bertscore_r":  R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


# CTGenAggregator
class CTGenAggregator(nn.Module):
    """
    Spatio-Temporal Aggregator: CTViT tokens → LLM-compatible visual tokens.

    (B, 2304, 512) → pool3d → (B, 256, 512) → MLP → (B, llm_dim)
    """

    def __init__(
        self,
        visual_dim:  int = 512,
        llm_dim:     int = 5376,
        t_patches:   int = 4,
        h_patches:   int = 24,
        w_patches:   int = 24,
        target_t:    int = 4,
        target_s:    int = 8,
    ):
        super().__init__()
        self.t = t_patches
        self.h = h_patches
        self.w = w_patches
        self.num_tokens = target_t * target_s * target_s

        self.pool = nn.AdaptiveAvgPool3d((target_t, target_s, target_s))
        self.proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, visual_dim * 4),
            nn.GELU(),
            nn.Linear(visual_dim * 4, llm_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # CTViT may return (B, N, D) or a higher-dim tensor such as
        # (B, T_patches, S_patches, D) — flatten everything except batch and
        # feature dim so downstream code always sees (B, N, D).
        if tokens.dim() != 3:
            tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        B, N, D = tokens.shape
        x = tokens.permute(0, 2, 1).reshape(B, D, self.t, self.h, self.w)
        x = self.pool(x).to(tokens.dtype)   # pool may upcast; restore input dtype
        x = x.flatten(2).permute(0, 2, 1)
        return self.proj(x)


def build_aggregator(cfg):
    """Build CTGenAggregator from config."""
    m = cfg.model
    return CTGenAggregator(
        visual_dim=m.visual_token_dim, llm_dim=m.llm_hidden_dim,
        t_patches=m.ctvit_t, h_patches=m.ctvit_h, w_patches=m.ctvit_w,
        target_t=m.target_t, target_s=m.target_s,
    )


def num_visual_tokens(cfg):
    return cfg.model.target_t * cfg.model.target_s * cfg.model.target_s


def _load_volume(pt_path: str) -> torch.Tensor:
    # Older .pt files saved by MONAI contain MetaTensors which require an
    # allowlist under PyTorch >=2.6's weights_only=True default.  Use False
    # (safe here: we created these files ourselves) and strip any MetaTensor
    # wrapper so downstream code receives a plain torch.Tensor.
    t = torch.load(pt_path, map_location="cpu", weights_only=False)
    if hasattr(t, "as_tensor"):
        t = t.as_tensor()
    return t.float()


def _build_id_map(pt_dir: str) -> dict:
    id_map = {}
    for p in glob.glob(os.path.join(pt_dir, "*.pt")):
        base = os.path.basename(p).replace(".pt", "")
        id_map[base] = p
        parts = base.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            id_map.setdefault(parts[0], p)
    return id_map


def _load_reports_df(raw_dir: str):
    csv_paths = glob.glob(os.path.join(raw_dir, "**", "*reports*.csv"), recursive=True)
    if not csv_paths:
        raise FileNotFoundError(f"No reports CSV under {raw_dir}")
    frames = [pd.read_csv(p) for p in csv_paths]
    df     = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    id_col = df.columns[0]
    # Case-insensitive substring match. Prefer Findings (most detailed text),
    # fall back to Impressions, Report, Text. CT-RATE uses e.g. "Findings_EN".
    rep_col = None
    for keyword in ["findings", "impression", "report", "text"]:
        for c in df.columns:
            if keyword in c.lower():
                rep_col = c
                break
        if rep_col is not None:
            break
    if rep_col is None:
        raise ValueError(
            f"No report column matched in {csv_paths[0]}. "
            f"Columns: {list(df.columns)}"
        )
    return df, id_col, rep_col


class CTReportDataset(Dataset):
    """Stage 1: pairs preprocessed CT tensors with raw report text."""

    def __init__(self, pt_dir, raw_dir, tokenizer, max_len=256, volume_names=None):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.id_map    = _build_id_map(pt_dir)

        df, id_col, self.rep_col = _load_reports_df(raw_dir)
        csv_ids = df[id_col].astype(str).str.replace(".nii.gz", "", regex=False)
        mask = csv_ids.isin(self.id_map)
        if volume_names is not None:
            mask = mask & csv_ids.isin(set(volume_names))
        df      = df[mask].copy().reset_index(drop=True)
        csv_ids = csv_ids[mask].reset_index(drop=True)
        df["_pt_path"] = csv_ids.map(self.id_map)
        self.df = df.dropna(subset=["_pt_path"]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row    = self.df.iloc[i]
        volume = _load_volume(row["_pt_path"])
        report = str(row[self.rep_col]) if pd.notna(row[self.rep_col]) else ""
        enc    = self.tokenizer(
            report, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        return {
            "volume":         volume,
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class CTInstructDataset(Dataset):
    """Stage 2: instruction-tuning with label masking."""

    def __init__(self, pt_dir, raw_dir, tokenizer, instruction, n_visual_tokens,
                 max_len=512, volume_names=None):
        self.tokenizer      = tokenizer
        self.max_len        = max_len
        self.instruction    = instruction
        self.n_visual_tokens = n_visual_tokens
        self.id_map         = _build_id_map(pt_dir)

        df, id_col, self.rep_col = _load_reports_df(raw_dir)
        csv_ids = df[id_col].astype(str).str.replace(".nii.gz", "", regex=False)
        mask = csv_ids.isin(self.id_map)
        if volume_names is not None:
            mask = mask & csv_ids.isin(set(volume_names))
        df      = df[mask].copy().reset_index(drop=True)
        csv_ids = csv_ids[mask].reset_index(drop=True)
        df["_pt_path"] = csv_ids.map(self.id_map)
        self.df = df.dropna(subset=["_pt_path"]).reset_index(drop=True)

        # Multimodal tokenizers (e.g. Gemma 3 4B+) return a BatchEncoding dict
        # from apply_chat_template even with return_tensors="pt"; text-only
        # tokenizers return a bare tensor. Handle both.
        prompt_out = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        prompt_ids = prompt_out if isinstance(prompt_out, torch.Tensor) else prompt_out["input_ids"]
        self.prompt_len = (
            prompt_ids.shape[1]
            + 1 + n_visual_tokens + 1  # <|visual_start|> + N + <|visual_end|>
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row    = self.df.iloc[i]
        volume = _load_volume(row["_pt_path"])
        report = str(row[self.rep_col]) if pd.notna(row[self.rep_col]) else ""

        full_text = self.tokenizer.apply_chat_template(
            [
                {"role": "user",      "content": self.instruction},
                {"role": "assistant", "content": report},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        enc       = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels    = input_ids.clone()
        labels[: self.prompt_len] = -100

        return {"volume": volume, "input_ids": input_ids, "labels": labels, "report": report}


def split_volumes(raw_dir, pt_dir, test_split=0.05, seed=42):
    """Deterministically split volume names into train+val and test sets."""
    id_map = _build_id_map(pt_dir)
    df, id_col, _ = _load_reports_df(raw_dir)
    csv_ids = (
        df[id_col].astype(str)
        .str.replace(".nii.gz", "", regex=False)
    )
    all_names = sorted(set(csv_ids) & set(id_map.keys()))

    gen     = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(all_names), generator=gen).tolist()
    n_test  = max(1, int(len(all_names) * test_split))

    test_names  = [all_names[i] for i in indices[:n_test]]
    train_names = [all_names[i] for i in indices[n_test:]]
    return train_names, test_names


def make_loaders(ds, batch_size=1, val_split=0.1, num_workers=None, seed=42):
    n_val   = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    nw = num_workers if num_workers is not None else min(os.cpu_count() or 4, 8)
    # persistent_workers and prefetch_factor require nw > 0
    pw = nw > 0
    pf = 2 if nw > 0 else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=pw, prefetch_factor=pf)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=pw, prefetch_factor=pf)
    # val_ds is returned so Stage 2 can build a non-distributed metric loader
    return train_loader, val_loader, val_ds, n_train, n_val



def load_visual_encoder(ct_clip_dir: str, weights_path: str) -> nn.Module:
    """Load CTViT from CT-CLIP weights, freeze it, keep float32."""
    sys.path.insert(0, os.path.join(ct_clip_dir, "CT_CLIP"))
    sys.path.insert(0, os.path.join(ct_clip_dir, "transformer_maskgit"))

    from ct_clip.ct_clip import CTCLIP
    from transformer_maskgit.ctvit import CTViT

    text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    text_encoder.resize_token_embeddings(30522)

    vision_transformer = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
        dim_head=32, heads=8,
    )
    wrapper = CTCLIP(
        image_encoder=vision_transformer,
        text_encoder=text_encoder,
        dim_image=294912, dim_text=768, dim_latent=512,
        extra_latent_projection=False, use_mlm=False,
        downsample_image_embeds=False, use_all_token_embeds=False,
    )
    wrapper.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True), strict=False
    )

    encoder = wrapper.visual_transformer.float()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    del wrapper
    return encoder


def load_llm_and_tokenizer(model_id: str, visual_start_token: str, visual_end_token: str):
    """
    Load LLM in bf16 (NO quantization — FSDP handles sharding).
    MedGemma 27B bf16 is ~54 GB; FSDP shards across 4-8 A100 80GB GPUs.
    """
    hf_token = os.environ.get("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(
        [visual_start_token, visual_end_token], special_tokens=True
    )

    # bf16 with no device_map — FSDP will shard and place the model
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        token=hf_token,
    )
    llm.resize_token_embeddings(len(tokenizer))
    llm.config.use_cache = False  # required for gradient checkpointing

    return llm, tokenizer


class MultimodalLLM(nn.Module):
    """
    Wraps the LLM so that visual token injection and embedding lookups happen
    inside a single forward() call.  Required for FSDP compatibility: with
    FULL_SHARD the embedding weight is sharded and only gathered inside an
    FSDP forward context — calling get_input_embeddings() outside forward()
    yields a 1-D shard that torch.nn.functional.embedding rejects.
    """

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, visual_tokens, input_ids,
                visual_start_id: int, visual_end_id: int,
                n_visual_tokens: int, labels=None):
        B   = visual_tokens.size(0)
        dev = visual_tokens.device
        emb = self.llm.get_input_embeddings()

        start_emb   = emb(torch.full((B, 1), visual_start_id, dtype=torch.long, device=dev))
        end_emb     = emb(torch.full((B, 1), visual_end_id,   dtype=torch.long, device=dev))
        text_embeds = emb(input_ids)

        embeds         = torch.cat([start_emb, visual_tokens, end_emb, text_embeds], dim=1)
        attn           = torch.ones(B, embeds.size(1), dtype=torch.long, device=dev)
        token_type_ids = torch.zeros(B, embeds.size(1), dtype=torch.long, device=dev)

        full_labels = None
        if labels is not None:
            prefix      = torch.full((B, 1 + n_visual_tokens + 1), -100, device=dev)
            full_labels = torch.cat([prefix, labels], dim=1)

        return self.llm(inputs_embeds=embeds, attention_mask=attn,
                        token_type_ids=token_type_ids, labels=full_labels)

    def generate(self, inputs_embeds, **gen_kwargs):
        """
        Run generation with all FSDP params gathered.

        Calling self.llm.generate() bypasses the root FSDP forward hook, so
        root-unit params (final norm, lm_head) are never all-gathered by the
        normal FSDP machinery.  summon_full_params(recurse=True) materialises
        every FSDP unit's params for the duration of the generate loop.
        _fsdp_handle is set by main() after accelerator.prepare().
        """
        fsdp_handle = getattr(self, '_fsdp_handle', None)
        ctx = (FSDP_class.summon_full_params(fsdp_handle, writeback=False, recurse=True)
               if fsdp_handle is not None else nullcontext())
        with ctx:
            return self.llm.generate(inputs_embeds=inputs_embeds, **gen_kwargs)

    def gradient_checkpointing_enable(self, **kw):
        self.llm.gradient_checkpointing_enable(**kw)

    def print_trainable_parameters(self):
        self.llm.print_trainable_parameters()

    @property
    def config(self):
        return self.llm.config


def _build_gen_embeds(fsdp_llm, vis_emb, instruct_ids, prefix_ids,
                      visual_start_id, visual_end_id, device):
    """
    Build inputs_embeds for generation.  Must be called with the FSDP-wrapped
    llm handle so summon_full_params can gather the (otherwise sharded)
    embedding weight on all ranks before the lookup.
    """
    inner = getattr(fsdp_llm, '_fsdp_wrapped_module', fsdp_llm)
    B     = vis_emb.size(0)
    ctx   = (FSDP_class.summon_full_params(fsdp_llm, writeback=False, recurse=False)
             if isinstance(fsdp_llm, FSDP_class) else nullcontext())
    with ctx:
        emb       = inner.llm.get_input_embeddings()
        start_emb = emb(torch.tensor([[visual_start_id]], dtype=torch.long, device=device))
        end_emb   = emb(torch.tensor([[visual_end_id]],   dtype=torch.long, device=device))
        embeds    = torch.cat(
            [start_emb, vis_emb, end_emb, emb(instruct_ids), emb(prefix_ids)], dim=1
        )
    # summon_full_params materialises weights in fp32; cast back to bf16 so
    # inputs_embeds matches the decoder layer weights during generation.
    return embeds.to(torch.bfloat16)


def train_stage1(cfg, accelerator, visual_encoder, projector, llm, tokenizer,
                 report_dataset, visual_start_id, visual_end_id):
    s1       = cfg.stage1
    ckpt_dir = os.path.join(cfg.paths.ckpt_dir, "stage1")
    nvt      = num_visual_tokens(cfg)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Freeze encoder + LLM (incl. LoRA — unfrozen later in Stage 2).
    # LoRA was already attached and gradient checkpointing already enabled
    # in main() before FSDP wrap.
    visual_encoder.eval()
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False
    projector.train()
    for p in projector.parameters():
        p.requires_grad = True

    train_loader, val_loader, _, n_train, n_val = make_loaders(
        report_dataset, batch_size=s1.batch_size,
        val_split=cfg.data.val_split, num_workers=cfg.data.num_workers,
    )
    optimizer = AdamW(projector.parameters(), lr=s1.lr, weight_decay=0.01)

    total_steps  = s1.epochs * (n_train // (s1.accum_steps * accelerator.num_processes))
    warmup_steps = max(1, total_steps // 10)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # projector + llm are already FSDP-wrapped by main(); only prepare the
    # new per-stage optimizer, scheduler and data loaders.
    optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        optimizer, train_loader, val_loader, scheduler,
    )

    if accelerator.is_main_process:
        wandb.init(
            project=cfg.wandb.project,
            name="stage1-alignment",
            mode="online" if os.environ.get("WANDB_API_KEY") else "offline",
            config={
                "stage": 1, "llm": cfg.model.llm_id,
                "visual_tokens": nvt,
                "epochs": s1.epochs, "lr": s1.lr,
                "accum_steps": s1.accum_steps,
                "num_gpus": accelerator.num_processes,
                "effective_batch": s1.batch_size * s1.accum_steps * accelerator.num_processes,
            },
        )
        print(f"Stage 1 | train={n_train}  val={n_val}  "
              f"steps={total_steps}  warmup={warmup_steps}")
        trainable = sum(p.numel() for p in projector.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable/1e6:.2f}M")

    pad_id      = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
    best_val    = float("inf")
    global_step = 0

    for epoch in range(1, s1.epochs + 1):
        projector.train()
        optimizer.zero_grad()
        running_loss = 0
        log_steps    = 0

        for batch_idx, batch in enumerate(train_loader):
            volume    = batch["volume"].to(dtype=torch.float32)
            input_ids = batch["input_ids"]

            with torch.no_grad():
                raw_tokens = visual_encoder(volume, return_encoded_tokens=True)

            visual_embeds = projector(raw_tokens.to(torch.bfloat16))

            s1_labels = input_ids.clone()
            s1_labels[input_ids == pad_id] = -100

            outputs = llm(visual_embeds, input_ids,
                          visual_start_id, visual_end_id, nvt,
                          labels=s1_labels)
            loss    = outputs.loss / s1.accum_steps

            accelerator.backward(loss)
            running_loss += loss.item()

            if (batch_idx + 1) % s1.accum_steps == 0:
                accelerator.clip_grad_norm_(projector.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                log_steps   += 1

                if accelerator.is_main_process and global_step % s1.log_every == 0:
                    avg = running_loss / log_steps
                    lr  = scheduler.get_last_lr()[0]
                    print(f"[S1 E{epoch}/{s1.epochs}] step={global_step} "
                          f"loss={avg:.4f}  lr={lr:.2e}")
                    wandb.log({"s1/train_loss": avg, "s1/lr": lr,
                               "s1/epoch": epoch, "s1/step": global_step})
                    running_loss = 0
                    log_steps    = 0

        # Validation
        projector.eval()
        val_loss = val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                volume    = batch["volume"].to(dtype=torch.float32)
                input_ids = batch["input_ids"]

                raw_tokens    = visual_encoder(volume, return_encoded_tokens=True)
                visual_embeds = projector(raw_tokens.to(torch.bfloat16))

                s1_labels = input_ids.clone()
                s1_labels[input_ids == pad_id] = -100

                out      = llm(visual_embeds, input_ids,
                               visual_start_id, visual_end_id, nvt,
                               labels=s1_labels)
                val_loss += out.loss.item()
                val_n   += 1

        val_loss /= max(val_n, 1)

        if accelerator.is_main_process:
            print(f"[S1 E{epoch}] val_loss={val_loss:.4f}")
            wandb.log({"s1/val_loss": val_loss, "s1/epoch": epoch,
                        "s1/step": global_step})
        if val_loss < best_val:
            best_val = val_loss
            accelerator.save_state(os.path.join(ckpt_dir, "best"))  # collective: all ranks
            if accelerator.is_main_process:
                print(f"  -> Best checkpoint (val_loss={best_val:.4f})")

    accelerator.wait_for_everyone()
    accelerator.save_state(os.path.join(ckpt_dir, "final"))  # collective: all ranks
    if accelerator.is_main_process:
        wandb.log({"s1/best_val_loss": best_val})
        wandb.finish()
        print("Stage 1 complete.")


# Stage 2: Instruction Tuning
def train_stage2(cfg, accelerator, visual_encoder, projector, llm, tokenizer,
                 instruct_dataset, visual_start_id, visual_end_id):
    s2       = cfg.stage2
    ckpt_dir = os.path.join(cfg.paths.ckpt_dir, "stage2")
    nvt      = num_visual_tokens(cfg)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Stage 1 just finished in-memory; the trained projector is live here.
    # LoRA was already attached to the LLM in main() (before FSDP wrap).
    # Unfreeze LoRA params; base model stays frozen.
    lora_params = [p for n, p in llm.named_parameters() if "lora_" in n.lower()]
    if not lora_params:
        raise RuntimeError(
            "No LoRA parameters found on llm — was LoRA applied in main()?"
        )
    for p in lora_params:
        p.requires_grad = True

    projector.train()
    for p in projector.parameters():
        p.requires_grad = True

    if accelerator.is_main_process:
        n_proj = sum(p.numel() for p in projector.parameters() if p.requires_grad)
        n_lora = sum(p.numel() for p in lora_params)
        print(f"Stage 2 trainable: projector={n_proj/1e6:.2f}M "
              f"| lora={n_lora/1e6:.2f}M")

    train_loader, val_loader, val_ds, n_train, n_val = make_loaders(
        instruct_dataset, batch_size=s2.batch_size,
        val_split=cfg.data.val_split, num_workers=cfg.data.num_workers,
    )

    optimizer = AdamW(
        [
            {"params": projector.parameters(), "lr": s2.lr},
            {"params": lora_params,            "lr": s2.lr * 0.1},
        ],
        weight_decay=0.01,
    )

    total_steps  = s2.epochs * (n_train // (s2.accum_steps * accelerator.num_processes))
    warmup_steps = max(1, total_steps // 10)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # projector + llm are already FSDP-wrapped; only prepare new opt/sched/loaders.
    optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        optimizer, train_loader, val_loader, scheduler,
    )

    if accelerator.is_main_process:
        if wandb.run is not None:
            wandb.finish()
        wandb.init(
            project=cfg.wandb.project,
            name="stage2-instruct-tuning",
            mode="online" if os.environ.get("WANDB_API_KEY") else "offline",
            config={
                "stage": 2, "llm": cfg.model.llm_id,
                "visual_tokens": nvt,
                "epochs": s2.epochs, "lr": s2.lr,
                "lora_r": cfg.model.lora_r, "lora_alpha": cfg.model.lora_alpha,
                "accum_steps": s2.accum_steps,
                "num_gpus": accelerator.num_processes,
                "effective_batch": s2.batch_size * s2.accum_steps * accelerator.num_processes,
            },
        )
        print(f"Stage 2 | train={n_train}  val={n_val}  "
              f"steps={total_steps}  warmup={warmup_steps}")

    best_val    = float("inf")
    global_step = 0

    for epoch in range(1, s2.epochs + 1):
        projector.train()
        llm.train()
        optimizer.zero_grad()
        running_loss = 0
        log_steps    = 0

        for batch_idx, batch in enumerate(train_loader):
            volume    = batch["volume"].to(dtype=torch.float32)
            input_ids = batch["input_ids"]
            labels    = batch["labels"]

            with torch.no_grad():
                raw_tokens = visual_encoder(volume, return_encoded_tokens=True)

            visual_embeds = projector(raw_tokens.to(torch.bfloat16))

            outputs = llm(visual_embeds, input_ids,
                          visual_start_id, visual_end_id, nvt,
                          labels=labels)
            loss    = outputs.loss / s2.accum_steps

            accelerator.backward(loss)
            running_loss += loss.item()

            if (batch_idx + 1) % s2.accum_steps == 0:
                accelerator.clip_grad_norm_(
                    list(projector.parameters()) +
                    [p for p in llm.parameters() if p.requires_grad],
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                log_steps   += 1

                if accelerator.is_main_process and global_step % s2.log_every == 0:
                    avg = running_loss / log_steps
                    lr  = scheduler.get_last_lr()[0]
                    print(f"[S2 E{epoch}/{s2.epochs}] step={global_step} "
                          f"loss={avg:.4f}  lr={lr:.2e}")
                    wandb.log({"s2/train_loss": avg, "s2/lr": lr,
                               "s2/epoch": epoch, "s2/step": global_step})
                    running_loss = 0
                    log_steps    = 0

        # Validation
        projector.eval()
        llm.eval()
        val_loss = val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                volume    = batch["volume"].to(dtype=torch.float32)
                input_ids = batch["input_ids"]
                labels    = batch["labels"]

                raw_tokens    = visual_encoder(volume, return_encoded_tokens=True)
                visual_embeds = projector(raw_tokens.to(torch.bfloat16))

                out      = llm(visual_embeds, input_ids,
                               visual_start_id, visual_end_id, nvt,
                               labels=labels)
                val_loss += out.loss.item()
                val_n   += 1

        val_loss /= max(val_n, 1)

        if accelerator.is_main_process:
            print(f"[S2 E{epoch}] val_loss={val_loss:.4f}")

        # ── Generation metrics on a fixed subset of val ────────────
        n_metric = getattr(cfg.eval, "val_metric_samples", 0)
        if n_metric > 0:
            n_metric = min(n_metric, len(val_ds))
            metric_subset = torch.utils.data.Subset(val_ds, list(range(n_metric)))
            # Non-distributed loader — all FSDP ranks iterate the same batches
            # so every collective forward/generate call stays in sync.
            metric_loader = DataLoader(metric_subset, batch_size=1, shuffle=False,
                                       num_workers=0)
            gen_texts = []
            ref_texts = []
            projector.eval()
            llm.eval()
            llm.config.use_cache = True   # faster autoregressive decoding

            _tmpl_out = tokenizer.apply_chat_template(
                [{"role": "user", "content": cfg.instruction}],
                add_generation_prompt=True, tokenize=True, return_tensors="pt",
            )
            instruct_ids_tmpl = (
                _tmpl_out if isinstance(_tmpl_out, torch.Tensor)
                else _tmpl_out["input_ids"]
            ).to(accelerator.device)
            prefix_ids = tokenizer(
                "\n**Findings:** ", return_tensors="pt", add_special_tokens=False,
            )["input_ids"].to(accelerator.device)

            # All FSDP ranks must participate in every forward/generate call.
            accelerator.wait_for_everyone()

            # Use a single top-level summon_full_params for the entire
            # generation block.  This avoids the fragile _fsdp_handle
            # mechanism inside MultimodalLLM.generate() and ensures ALL
            # FSDP-sharded params (including root-level norm / lm_head)
            # are fully gathered for the duration of autoregressive
            # decoding, which bypasses FSDP forward hooks.
            _gen_ctx = (FSDP_class.summon_full_params(llm, writeback=False, recurse=True)
                        if isinstance(llm, FSDP_class) else nullcontext())

            with _gen_ctx, torch.no_grad():
                unwrapped_llm = accelerator.unwrap_model(llm)
                emb_fn = unwrapped_llm.llm.get_input_embeddings()

                for mbatch in metric_loader:
                    volume   = mbatch["volume"].to(accelerator.device, dtype=torch.float32)
                    ref_text = mbatch["report"][0]

                    raw_tok = visual_encoder(volume, return_encoded_tokens=True)
                    vis_emb = projector(raw_tok.to(torch.bfloat16))

                    # Build generation embeddings directly — no nested
                    # summon_full_params needed since the outer context
                    # already gathered all params.
                    dev = accelerator.device
                    start_emb = emb_fn(torch.tensor([[visual_start_id]], dtype=torch.long, device=dev))
                    end_emb   = emb_fn(torch.tensor([[visual_end_id]],   dtype=torch.long, device=dev))
                    input_embeds = torch.cat(
                        [start_emb, vis_emb, end_emb, emb_fn(instruct_ids_tmpl), emb_fn(prefix_ids)], dim=1
                    ).to(torch.bfloat16)

                    out_ids = unwrapped_llm.llm.generate(
                        inputs_embeds=input_embeds,
                        max_new_tokens=cfg.eval.max_tokens,
                        do_sample=False,
                        temperature=1.0,
                        repetition_penalty=1.2,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                    if accelerator.is_main_process:
                        gen_texts.append(tokenizer.decode(out_ids[0], skip_special_tokens=True))
                        ref_texts.append(ref_text)

            accelerator.wait_for_everyone()
            llm.config.use_cache = False   # restore for training

            if accelerator.is_main_process and gen_texts:
                gen_metrics = compute_metrics(gen_texts, ref_texts)
                print(
                    f"[S2 E{epoch}] val gen ({len(gen_texts)} samples) | "
                    f"BLEU-1={gen_metrics['bleu_1']:.3f}  "
                    f"ROUGE-1={gen_metrics['rouge_1']:.3f}  "
                    f"METEOR={gen_metrics['meteor']:.3f}  "
                    f"BERTScore-F1={gen_metrics['bertscore_f1']:.3f}"
                )
                wandb.log({
                    "s2/val_bleu_1":       gen_metrics["bleu_1"],
                    "s2/val_bleu_4":       gen_metrics["bleu_4"],
                    "s2/val_meteor":       gen_metrics["meteor"],
                    "s2/val_rouge_1":      gen_metrics["rouge_1"],
                    "s2/val_rouge_2":      gen_metrics["rouge_2"],
                    "s2/val_rouge_L":      gen_metrics["rouge_L"],
                    "s2/val_bertscore_p":  gen_metrics["bertscore_p"],
                    "s2/val_bertscore_r":  gen_metrics["bertscore_r"],
                    "s2/val_bertscore_f1": gen_metrics["bertscore_f1"],
                    "s2/epoch": epoch, "s2/step": global_step,
                })

        if accelerator.is_main_process:
            wandb.log({"s2/val_loss": val_loss, "s2/epoch": epoch,
                        "s2/step": global_step})
        if val_loss < best_val:
            best_val = val_loss
            accelerator.save_state(os.path.join(ckpt_dir, "best"))  # collective: all ranks
            if accelerator.is_main_process:
                print(f"  -> Best S2 checkpoint (val_loss={best_val:.4f})")

    accelerator.wait_for_everyone()
    accelerator.save_state(os.path.join(ckpt_dir, "final"))  # collective: all ranks
    if accelerator.is_main_process:
        wandb.log({"s2/best_val_loss": best_val})
        wandb.finish()
        print("Stage 2 complete.")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    os.makedirs(cfg.paths.ckpt_dir, exist_ok=True)
    # Persist the fully-resolved config so evaluate.py / inference.py can
    # reconstruct identical model architecture regardless of CLI overrides.
    OmegaConf.save(cfg, os.path.join(cfg.paths.ckpt_dir, "train_config.yaml"))
    nvt = num_visual_tokens(cfg)

    # Fail fast if HF_TOKEN is absent — the LLM load will 401 anyway, but
    # this surfaces the problem before wasting time on CTViT loading.
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set. MedGemma is a gated model.\n"
            "  export HF_TOKEN=hf_..."
        )

    # Sanitize WANDB_MODE.  On Colab the env var is sometimes set to an API
    # key string by mistake, which causes wandb's pydantic Settings to raise a
    # ValidationError before any training starts.
    _valid_wandb_modes = {"online", "offline", "shared", "disabled", "dryrun", "run"}
    if os.environ.get("WANDB_MODE", "") not in _valid_wandb_modes:
        # Use online if we have an API key, otherwise offline.
        os.environ["WANDB_MODE"] = (
            "online" if os.environ.get("WANDB_API_KEY") else "offline"
        )

    # Explicit login so the API key is picked up in Colab subprocess contexts.
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key, relogin=True)

    # Deterministic init across FSDP ranks. resize_token_embeddings adds
    # randomly-initialised rows for the visual-start/end tokens; without a
    # fixed seed, each rank would get different values and diverge.
    # sync_module_states on the FSDP plugin additionally broadcasts rank-0
    # weights at wrap time as a belt-and-braces safeguard.
    torch.manual_seed(42)

    # FSDP wrap policy — Gemma3DecoderLayer MUST be importable, otherwise
    # the entire 27B model collapses into a single FSDP unit and sharding
    # breaks. transformers>=4.52.0 is required (pinned in setup.sh).
    try:
        from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
    except ImportError as e:
        raise ImportError(
            "Gemma3DecoderLayer could not be imported from transformers. "
            "FSDP requires per-layer wrapping to shard MedGemma 27B across GPUs. "
            "Upgrade to transformers>=4.52.0:  pip install 'transformers>=4.52.0'"
        ) from e
    wrap_classes = {Gemma3DecoderLayer}

    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=wrap_classes,
        ),
        use_orig_params=True,
        sync_module_states=True,       # broadcast rank-0 weights during wrap
        cpu_ram_efficient_loading=True, # only rank-0 loads weights; others get shards
        cpu_offload=False,
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        fsdp_plugin=fsdp_plugin,
        project_dir=cfg.paths.ckpt_dir,
    )

    if accelerator.is_main_process:
        print(f"Accelerator | device={accelerator.device} | "
              f"procs={accelerator.num_processes} | mixed_precision=bf16")
        print(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if accelerator.is_main_process:
        print("\nLoading CTViT...")
    visual_encoder = load_visual_encoder(cfg.paths.ct_clip_dir, cfg.paths.ct_clip_weights)
    visual_encoder = visual_encoder.to(accelerator.device)

    if accelerator.is_main_process:
        print("Loading LLM (bf16, no quantization)...")
    llm, tokenizer = load_llm_and_tokenizer(
        cfg.model.llm_id, cfg.tokens.visual_start, cfg.tokens.visual_end,
    )

    # Attach LoRA UPFRONT, before FSDP wrap. In Stage 1 all LoRA params are
    # frozen (only projector trains); Stage 2 unfreezes them. Doing this
    # here means we only ever call accelerator.prepare() on the LLM once.
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=list(cfg.model.lora_target_modules),
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
    )
    llm = get_peft_model(llm, lora_cfg)
    # PEFT initialises LoRA weights in float32; cast to bf16 to match the
    # frozen base model — FSDP requires uniform dtype within each sharded unit.
    for param in llm.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
    llm.gradient_checkpointing_enable()
    if accelerator.is_main_process:
        llm.print_trainable_parameters()

    # Wrap in MultimodalLLM so embedding lookups happen inside forward(),
    # which is the only FSDP-safe context for accessing sharded weights.
    llm = MultimodalLLM(llm)

    visual_start_id = tokenizer.convert_tokens_to_ids(cfg.tokens.visual_start)
    visual_end_id   = tokenizer.convert_tokens_to_ids(cfg.tokens.visual_end)

    if accelerator.is_main_process:
        print("Building CTGenAggregator...")
    projector = build_aggregator(cfg).to(torch.bfloat16)

    # SINGLE FSDP wrap for projector + llm. Never re-prepare these models.
    # Per-stage optimizers/schedulers/loaders are prepared inside the stages.
    projector, llm = accelerator.prepare(projector, llm)

    # Give the inner MultimodalLLM a reference to the FSDP wrapper so its
    # generate() method can call summon_full_params on all ranks.
    # Use accelerator.unwrap_model for robustness across Accelerate versions
    # (the wrapper type may not always be FSDP_class directly).
    _inner_llm = accelerator.unwrap_model(llm)
    _inner_llm._fsdp_handle = llm if isinstance(llm, FSDP_class) else None

    if accelerator.is_main_process:
        print("\nSplitting volumes...")
    train_names, test_names = split_volumes(
        cfg.paths.raw_dir, cfg.paths.pt_dir, test_split=cfg.data.test_split,
    )

    test_split_file = os.path.join(cfg.paths.ckpt_dir, "test_volumes.json")
    if accelerator.is_main_process:
        with open(test_split_file, "w") as f:
            json.dump(test_names, f)
        print(f"  Train+val: {len(train_names)}  |  Test (held-out): {len(test_names)}")
        print(f"  Test split saved: {test_split_file}")

    if accelerator.is_main_process:
        print("\nLoading datasets...")
    report_dataset = CTReportDataset(
        cfg.paths.pt_dir, cfg.paths.raw_dir, tokenizer,
        volume_names=train_names,
    )
    instruct_dataset = CTInstructDataset(
        cfg.paths.pt_dir, cfg.paths.raw_dir, tokenizer,
        instruction=cfg.instruction, n_visual_tokens=nvt,
        volume_names=train_names,
    )

    if accelerator.is_main_process:
        print(f"  Stage 1 dataset: {len(report_dataset)} pairs")
        print(f"  Stage 2 dataset: {len(instruct_dataset)} pairs")

    # Stage 1: projector-only alignment (LoRA frozen). Skip if epochs=0.
    if cfg.stage1.epochs > 0:
        train_stage1(
            cfg, accelerator, visual_encoder, projector, llm, tokenizer,
            report_dataset, visual_start_id, visual_end_id,
        )
    else:
        # Try to resume from the best stage 1 checkpoint so stage 2 starts
        # from trained projector weights rather than random initialisation.
        for candidate in ["best", "final"]:
            ckpt = os.path.join(cfg.paths.ckpt_dir, "stage1", candidate)
            if os.path.isdir(ckpt):
                if accelerator.is_main_process:
                    print(f"Stage 1 skipped — loading checkpoint: {ckpt}")
                accelerator.load_state(ckpt)  # collective: all ranks
                break
        else:
            if accelerator.is_main_process:
                print("Stage 1 skipped — no checkpoint found, projector starts random.")

    # Stage 2: unfreezes LoRA; trains projector + LoRA.
    train_stage2(
        cfg, accelerator, visual_encoder, projector, llm, tokenizer,
        instruct_dataset, visual_start_id, visual_end_id,
    )

    if accelerator.is_main_process:
        print("\nAll training complete.")
        print("Run evaluation:  python evaluate.py")


if __name__ == "__main__":
    main()
