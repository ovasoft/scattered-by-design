# Notice: vendored copy of locuslab/wanda

This directory is a vendored copy of the official Wanda implementation
(Sun, Liu, Bair, Kolter, "A Simple and Effective Pruning Approach for Large
Language Models", ICLR 2024).

| | |
|---|---|
| Upstream URL | https://github.com/locuslab/wanda |
| Upstream commit | `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6` ("add dense ft", committed 2023-11-03T01:37:59-04:00) |
| Vendored on | 2026-10-18 |
| Licence | MIT, see [`LICENSE`](LICENSE) (retained verbatim; Copyright (c) 2023 CMU Locus Lab) |

The tree is redistributed under its original MIT licence **with local
modifications**. The modifications are listed in
[`MODIFICATIONS.md`](MODIFICATIONS.md), marked in the source with
`# MODIFIED (scattered-by-design)`, and given as unified diffs against the
upstream commit above in [`../../patches/`](../../patches/).

Everything else in this directory is upstream code, unchanged, apart from
these exclusions: the upstream `.git` directory, `__pycache__/` directories,
and a local scratch file (`lib/prune_original.py`) that was never part of
upstream.

Wanda's SparseGPT path (`lib/sparsegpt.py`) is itself derived from
IST-DASLab/sparsegpt (Frantar & Alistarh, 2023), as in upstream.
