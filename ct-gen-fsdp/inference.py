#!/usr/bin/env python3
"""
CT-Gen FSDP — Single-GPU inference demo.

Usage:
    python inference.py [--volume /path/to/volume.pt]

Loads the best Stage 2 checkpoint and generates a report.
"""
import os
import sys
import glob
import argparse

import torch
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

from train import (
    load_config,
    build_aggregator, num_visual_tokens,
    load_visual_encoder, load_llm_and_tokenizer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=str, default=None,
                        help="Path to a preprocessed .pt volume")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max tokens to generate (default: from config)")
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
    max_tokens = args.max_tokens or cfg.inference.max_tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading CTViT...")
    visual_encoder = load_visual_encoder(
        cfg.paths.ct_clip_dir, cfg.paths.ct_clip_weights,
    ).to(device)

    print("Loading LLM...")
    llm, tokenizer = load_llm_and_tokenizer(
        cfg.model.llm_id, cfg.tokens.visual_start, cfg.tokens.visual_end,
    )
    llm.config.use_cache = True   # re-enable KV cache (disabled only for training)
    
    # MedGemma 27B bf16 is ~54 GB — use device_map for inference to split
    # across GPUs if needed (single A100 80GB is tight but workable)
    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model, infer_auto_device_map
        device_map = infer_auto_device_map(llm, max_memory={
            i: "75GiB" for i in range(torch.cuda.device_count())
        })
        llm = dispatch_model(llm, device_map=device_map)
        device = llm.device if hasattr(llm, 'device') else "cuda:0"
    else:
        llm = llm.to(device)

    visual_start_id = tokenizer.convert_tokens_to_ids(cfg.tokens.visual_start)
    visual_end_id   = tokenizer.convert_tokens_to_ids(cfg.tokens.visual_end)

    print("Loading CTGenAggregator...")
    projector = build_aggregator(cfg).to(device)

    # New format (save_stage2_checkpoint): projector.pt + lora/ dir.
    # Legacy accelerator.save_state layout also supported as fallback.
    proj_ckpt = os.path.join(cfg.paths.ckpt_dir, "stage2", "best")
    if os.path.exists(proj_ckpt):
        from safetensors.torch import load_file as _sf_load

        proj_candidates = [
            os.path.join(proj_ckpt, "projector.pt"),
            os.path.join(proj_ckpt, "pytorch_model.bin"),
            os.path.join(proj_ckpt, "model.safetensors"),
        ]
        proj_file = next((p for p in proj_candidates if os.path.exists(p)), None)
        if proj_file:
            _sd = (_sf_load(proj_file, device=str(device))
                   if proj_file.endswith(".safetensors")
                   else torch.load(proj_file, map_location=device, weights_only=True))
            projector.load_state_dict(_sd, strict=False)
            print(f"Loaded projector from {proj_file}")

        lora_dir_new    = os.path.join(proj_ckpt, "lora")
        lora_dir_legacy = os.path.join(proj_ckpt, "lora_adapters")
        llm_sf          = os.path.join(proj_ckpt, "model_1.safetensors")
        if os.path.isdir(lora_dir_new):
            llm = PeftModel.from_pretrained(llm, lora_dir_new)
            print(f"Loaded LoRA from {lora_dir_new}")
        elif os.path.isdir(lora_dir_legacy):
            llm = PeftModel.from_pretrained(llm, lora_dir_legacy)
            print(f"Loaded LoRA from {lora_dir_legacy}")
        elif os.path.exists(llm_sf):
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.model.lora_r,
                lora_alpha=cfg.model.lora_alpha,
                target_modules=list(cfg.model.lora_target_modules),
                lora_dropout=0.0,
                bias="none",
            )
            llm = get_peft_model(llm, lora_cfg)
            _sd = _sf_load(llm_sf, device=str(device))
            llm.load_state_dict(_sd, strict=False)
            print(f"Loaded LLM+LoRA from {llm_sf}")

    projector.to(torch.bfloat16).eval()
    llm.eval()


    if args.volume:
        vol_path = args.volume
    else:
        pt_files = sorted(glob.glob(os.path.join(cfg.paths.pt_dir, "*.pt")))
        if not pt_files:
            print("No .pt volumes found. Run download_and_preprocess.py first.")
            sys.exit(1)
        vol_path = pt_files[-1]

    print(f"\nInference on: {vol_path}")
    volume = torch.load(vol_path, map_location="cpu", weights_only=True).float()
    volume = volume.unsqueeze(0).to(device)  # (1, 1, 40, 480, 480)

    with torch.no_grad():
        raw_tokens    = visual_encoder(volume, return_encoded_tokens=True)
        visual_embeds = projector(raw_tokens.to(torch.bfloat16))

    emb = llm.get_input_embeddings()
    # Multimodal tokenizers return BatchEncoding from apply_chat_template;
    # text-only tokenizers return a bare tensor.
    _out = tokenizer.apply_chat_template(
        [{"role": "user", "content": cfg.instruction}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    instruct_ids = (_out if isinstance(_out, torch.Tensor) else _out["input_ids"]).to(device)

    start_emb    = emb(torch.tensor([[visual_start_id]], device=device))
    end_emb      = emb(torch.tensor([[visual_end_id]],   device=device))
    instruct_emb = emb(instruct_ids)

    prefix_text = "\n**Findings:** "
    prefix_ids  = tokenizer(prefix_text, return_tensors="pt",
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

    report = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print("\n=== Generated Report ===")
    print(report)


if __name__ == "__main__":
    main()
