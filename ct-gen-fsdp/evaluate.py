#!/usr/bin/env python3
"""
CT-Gen FSDP — Evaluation on held-out test set.

Generates reports for test volumes and computes standard radiology report
generation metrics: BLEU, ROUGE, METEOR, BERTScore.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint stage2/best --max_tokens 512
    python evaluate.py --output results/eval_run1.json
"""
import os
import sys
import json
import glob
import argparse

import torch
import pandas as pd
from tqdm.auto import tqdm
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

from train import (
    load_config,
    compute_metrics,
    CTGenAggregator, build_aggregator, num_visual_tokens,
    load_visual_encoder, load_llm_and_tokenizer,
    _load_reports_df, _build_id_map, split_volumes,
)

def generate_report(
    volume_tensor, visual_encoder, projector, llm, tokenizer,
    visual_start_id, visual_end_id, instruction, device, max_tokens=512,
):
    """Generate a radiology report for a single preprocessed volume."""
    volume = volume_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        raw_tokens    = visual_encoder(volume, return_encoded_tokens=True)
        visual_embeds = projector(raw_tokens.to(torch.bfloat16))

    emb = llm.get_input_embeddings()
    # Multimodal tokenizers return BatchEncoding from apply_chat_template;
    # text-only tokenizers return a bare tensor.
    _out = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    instruct_ids = (_out if isinstance(_out, torch.Tensor) else _out["input_ids"]).to(device)

    start_emb    = emb(torch.tensor([[visual_start_id]], device=device))
    end_emb      = emb(torch.tensor([[visual_end_id]],   device=device))
    instruct_emb = emb(instruct_ids)

    prefix_ids  = tokenizer("\n**Findings:** ", return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(device)
    prefix_emb  = emb(prefix_ids)

    input_embeds = torch.cat(
        [start_emb, visual_embeds, end_emb, instruct_emb, prefix_emb], dim=1
    )

    with torch.no_grad():
        out_ids = llm.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CT-Gen on held-out test set")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint subdir under $CKPT_DIR (default: from config)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max tokens to generate (default: from config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test samples (for quick checks)")
    parser.add_argument("--llm_id", type=str, default=None,
                        help="Override LLM model ID (e.g. google/gemma-3-4b-it)")
    parser.add_argument("--llm_hidden_dim", type=int, default=None,
                        help="Override LLM hidden dim (e.g. 2560 for 4B, 5376 for 27B)")
    args = parser.parse_args()

    overrides = []
    if args.llm_id:
        overrides.append(f"model.llm_id={args.llm_id}")
    if args.llm_hidden_dim:
        overrides.append(f"model.llm_hidden_dim={args.llm_hidden_dim}")
    cfg = load_config(overrides or None)
    checkpoint  = args.checkpoint or cfg.eval.checkpoint
    max_tokens  = args.max_tokens or cfg.eval.max_tokens
    output_path = args.output or os.path.join(cfg.paths.ckpt_dir, "eval_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test volume list
    test_split_file = os.path.join(cfg.paths.ckpt_dir, "test_volumes.json")
    if os.path.exists(test_split_file):
        with open(test_split_file) as f:
            test_names = json.load(f)
        print(f"Loaded {len(test_names)} test volumes from {test_split_file}")
    else:
        print(f"No test split file found at {test_split_file}")
        print("Generating split with same seed as training...")
        _, test_names = split_volumes(
            cfg.paths.raw_dir, cfg.paths.pt_dir, test_split=cfg.data.test_split,
        )
        print(f"  Test volumes: {len(test_names)}")

    if args.limit:
        test_names = test_names[:args.limit]
        print(f"  Limited to {args.limit} samples")

    # Load ground truth reports
    df, id_col, rep_col = _load_reports_df(cfg.paths.raw_dir)
    csv_ids = df[id_col].astype(str).str.replace(".nii.gz", "", regex=False)
    df["_vol_name"] = csv_ids
    report_lookup = dict(zip(df["_vol_name"], df[rep_col].astype(str)))

    # Resolve test volumes to .pt paths
    id_map = _build_id_map(cfg.paths.pt_dir)
    test_items = []
    for vn in test_names:
        if vn in id_map and vn in report_lookup:
            test_items.append((vn, id_map[vn], report_lookup[vn]))

    if not test_items:
        print("ERROR: No test volumes found on disk. Run download_and_preprocess.py first.")
        sys.exit(1)
    print(f"Test samples with both volume and report: {len(test_items)}")

    # Load models
    print("\nLoading CTViT...")
    visual_encoder = load_visual_encoder(
        cfg.paths.ct_clip_dir, cfg.paths.ct_clip_weights,
    ).to(device)

    print("Loading LLM...")
    llm, tokenizer = load_llm_and_tokenizer(
        cfg.model.llm_id, cfg.tokens.visual_start, cfg.tokens.visual_end,
    )
    llm.config.use_cache = True   # re-enable KV cache (disabled only for training)

    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model, infer_auto_device_map
        device_map = infer_auto_device_map(llm, max_memory={
            i: "75GiB" for i in range(torch.cuda.device_count())
        })
        llm = dispatch_model(llm, device_map=device_map)
    else:
        llm = llm.to(device)

    visual_start_id = tokenizer.convert_tokens_to_ids(cfg.tokens.visual_start)
    visual_end_id   = tokenizer.convert_tokens_to_ids(cfg.tokens.visual_end)

    print("Loading CTGenAggregator...")
    projector = build_aggregator(cfg).to(device)

    # Load checkpoint
    # accelerate save_state writes:
    #   model.safetensors   → projector (prepared first)
    #   model_1.safetensors → LLM + LoRA (prepared second)
    ckpt_path = os.path.join(cfg.paths.ckpt_dir, checkpoint)
    if os.path.exists(ckpt_path):
        from safetensors.torch import load_file as _sf_load

        # ── Projector ──────────────────────────────────────────────
        proj_file = os.path.join(ckpt_path, "pytorch_model.bin")
        if not os.path.exists(proj_file):
            proj_file = os.path.join(ckpt_path, "model.safetensors")
        if os.path.exists(proj_file):
            _sd = (_sf_load(proj_file, device=str(device))
                   if proj_file.endswith(".safetensors")
                   else torch.load(proj_file, map_location=device, weights_only=True))
            projector.load_state_dict(_sd, strict=False)
            print(f"Loaded projector from {ckpt_path}")

        # ── LLM + LoRA ─────────────────────────────────────────────
        llm_sf = os.path.join(ckpt_path, "model_1.safetensors")
        if os.path.exists(llm_sf):
            # Attach LoRA with same config as training, then load state dict.
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.model.lora_r,
                lora_alpha=cfg.model.lora_alpha,
                target_modules=list(cfg.model.lora_target_modules),
                lora_dropout=0.0,   # no dropout at inference
                bias="none",
            )
            llm = get_peft_model(llm, lora_cfg)
            _sd = _sf_load(llm_sf, device=str(device))
            llm.load_state_dict(_sd, strict=False)
            print(f"Loaded LLM+LoRA from {llm_sf}")
        elif os.path.exists(os.path.join(ckpt_path, "lora_adapters")):
            lora_dir = os.path.join(ckpt_path, "lora_adapters")
            llm = PeftModel.from_pretrained(llm, lora_dir)
            print(f"Loaded LoRA from {lora_dir}")
    else:
        print(f"WARNING: Checkpoint not found at {ckpt_path}, using untrained model")

    projector.to(torch.bfloat16).eval()
    llm.eval()

    # Generate reports
    print(f"\nGenerating reports for {len(test_items)} test volumes "
          f"(max_tokens={max_tokens})...")

    generated  = []
    references = []
    per_sample = []

    for vn, pt_path, ref_report in tqdm(test_items, desc="Evaluating"):
        volume = torch.load(pt_path, map_location="cpu", weights_only=True).float()

        gen_report = generate_report(
            volume, visual_encoder, projector, llm, tokenizer,
            visual_start_id, visual_end_id, cfg.instruction, device,
            max_tokens=max_tokens,
        )

        generated.append(gen_report)
        references.append(ref_report)
        per_sample.append({
            "volume": vn,
            "generated": gen_report,
            "reference": ref_report,
        })

    print("\nComputing metrics...")
    metrics = compute_metrics(generated, references)

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Samples:       {metrics['n_samples']}")
    print(f"  Scored:        {metrics['n_scored']}")
    print()
    print(f"  BLEU-1:        {metrics['bleu_1']:.4f}")
    print(f"  BLEU-4:        {metrics['bleu_4']:.4f}")
    print(f"  METEOR:        {metrics['meteor']:.4f}")
    print()
    print(f"  ROUGE-1:       {metrics['rouge_1']:.4f}")
    print(f"  ROUGE-2:       {metrics['rouge_2']:.4f}")
    print(f"  ROUGE-L:       {metrics['rouge_L']:.4f}")
    print()
    print(f"  BERTScore P:   {metrics['bertscore_p']:.4f}")
    print(f"  BERTScore R:   {metrics['bertscore_r']:.4f}")
    print(f"  BERTScore F1:  {metrics['bertscore_f1']:.4f}")
    print("=" * 50)

    output = {
        "checkpoint": checkpoint,
        "max_tokens": max_tokens,
        "metrics": metrics,
        "samples": per_sample,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
