# Local modifications to the vendored Wanda tree

Upstream commit: `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6` (see `NOTICE.md`).

There are exactly two functional modifications. Both are needed only because
the pinned library versions (`datasets==3.5.0`, `transformers==4.57.1`) are
newer than the versions the 2023 repository targets. Neither one changes the
pruning algorithm. Both are backward-compatible with the older libraries.

Each patch site in the source is preceded by the comment

    # MODIFIED (scattered-by-design): see third_party/wanda/MODIFICATIONS.md

The exact diffs against upstream are in `patches/` at the repository root.
They document the changes and are not a build step: the tree here already has
them applied. Applying both patches to a clean checkout of the upstream
commit reproduces this tree byte for byte, apart from the marker comments.

---

## 1. `lib/data.py`: C4 builder config removed from `load_dataset`

- **Patch:** `patches/0001-datasets-c4-config.patch`
- **Change:** in `get_c4`, both calls
  `load_dataset('allenai/c4', 'allenai--c4', data_files=..., split=...)`
  drop the second positional argument (`'allenai--c4'`).
- **Reason:** recent versions of `datasets` no longer have the legacy
  `'allenai--c4'` builder config, so the call raises. The `data_files=`
  argument already picks out the exact shards upstream used
  (`en/c4-train.00000-of-01024.json.gz` and
  `en/c4-validation.00000-of-00008.json.gz`).
- **Upstream behaviour restored:** the same two C4 shards are loaded, and the
  calibration samples are drawn with the same `random.seed(seed)` procedure.

## 2. `lib/prune.py` (LLaMA path only): forward `position_embeddings` to standalone decoder layers

- **Patch:** `patches/0002-transformers-rotary-position-embeddings.patch`
- **Reason:** recent versions of `transformers` compute LLaMA rotary
  embeddings once at model level and pass `position_embeddings=(cos, sin)` to
  each decoder layer. Wanda calls decoder layers standalone (block by block,
  on cached activations), so attention receives `None` and fails with
  `TypeError: cannot unpack non-iterable NoneType object`.
- **Change:**
  - Two helpers, `_layer_kwargs(attention_mask, position_ids, position_embeddings)`
    and `_move(obj, dev)`, near the top of the file.
  - Each `Catcher` (in `prepare_calibration_input`, `prune_sparsegpt`,
    `prune_ablate`) also records `kwargs.get('position_embeddings', None)`.
  - `prepare_calibration_input` returns it as a fifth value. The caller
    (`prune_wanda`) was updated to match.
  - Every standalone `layer(...)` call in `prune_wanda`, `prune_sparsegpt`
    and `prune_ablate` now passes `**_layer_kwargs(...)`.
  - The multi-GPU `hf_device_map` branch moves `position_embeddings` (a
    tuple) to the target device with `_move`. It also tolerates a `None`
    attention mask.
- **Backward compatibility:** `position_embeddings` is forwarded only when
  the captured value is not `None`. Under older `transformers`, the kwarg is
  never passed, so the layer computes rotary embeddings internally exactly as
  upstream expects.
- **Upstream behaviour restored:** the LLaMA pruning loop runs as in upstream.
  Masks, the SparseGPT update and the block-sequential propagation are
  unchanged.
- **Incidental:** the patch also strips trailing whitespace on a few
  untouched lines of `lib/prune.py`. This shows up in the diff but has no
  effect.

`lib/prune_opt.py` (the OPT path) is unmodified. OPT decoder layers in the
pinned `transformers` still accept being called standalone.
