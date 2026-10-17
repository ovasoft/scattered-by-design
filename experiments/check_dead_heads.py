import sys, types, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "./wanda")
from lib.prune_opt import prune_wanda
from lib.data import get_loaders

model_id = "facebook/opt-125m"
tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                             device_map="cuda")
model.eval()
model.seqlen = 2048

ns = types.SimpleNamespace(nsamples=128, seed=0, sparsity_ratio=0.5,
                           use_variant=False, cache_dir="llm_weights",
                           model=model_id)
prune_wanda(ns, model, tok, torch.device("cuda"))

for blk in [0, 1, 3]:
    W = model.model.decoder.layers[blk].self_attn.out_proj.weight
    W = W.detach().float().cpu().numpy()
    dead = (np.abs(W) < 1e-6).mean(axis=0) >= 0.99
    per_head = dead.reshape(12, 64).sum(axis=1)     # 12 heads x 64 dims
    print(f"layers.{blk}.out_proj: {dead.sum()} dead cols, per head: {per_head}")