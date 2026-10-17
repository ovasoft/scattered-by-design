"""
Dead attention heads behind the early-block out_proj anomaly (OPT-125M).

Prunes OPT-125M whole-model with the official Wanda code (50%, 128 C4
samples, seed 0) and counts, for out_proj of blocks 0, 1 and 3, the input
columns that are >= ROW_THRESH zero, grouped by attention head (12 heads x
64 dims). Whole input-column blocks dying together = dead heads. This is the
mechanism the paper gives for the out_proj exception.

Usage:
  python experiments/check_dead_heads.py [--blocks 0 1 3]
Prints to stdout; writes nothing.
"""
import os, sys, types, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning_transforms.config import ZERO_THRESH, ROW_THRESH, N_CALIB, DEFAULT_WANDA_REPO
from pruning_transforms.pruning import import_official_pruners

parser = argparse.ArgumentParser()
parser.add_argument("--wanda_repo", default=str(DEFAULT_WANDA_REPO))
parser.add_argument("--blocks", type=int, nargs="+", default=[0, 1, 3])
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "facebook/opt-125m"
prune_wanda = import_official_pruners("opt", args.wanda_repo)["prune_wanda"]

tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                             device_map=DEVICE)
model.eval()
model.seqlen = 2048

ns = types.SimpleNamespace(nsamples=N_CALIB, seed=0, sparsity_ratio=0.5,
                           use_variant=False, cache_dir="llm_weights",
                           model=model_id)
prune_wanda(ns, model, tok, torch.device(DEVICE))

n_heads = model.config.num_attention_heads
for blk in args.blocks:
    W = model.model.decoder.layers[blk].self_attn.out_proj.weight
    W = W.detach().float().cpu().numpy()
    dead = (np.abs(W) < ZERO_THRESH).mean(axis=0) >= ROW_THRESH
    per_head = dead.reshape(n_heads, -1).sum(axis=1)     # 12 heads x 64 dims
    print(f"layers.{blk}.out_proj: {dead.sum()} dead cols, per head: {per_head}")
