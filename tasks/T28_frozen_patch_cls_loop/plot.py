"""T28 results: CLS-only loop (T28-a) vs T11-d.

1. FLOPs of ONE loop for both models: counted with torch's FlopCounterMode on one call of the looped block with a
   (1, 197, 192) input, and by formula (also the CLS-only loop with the patch keys/values cached, which our code does
   not do). Writes results/flops.json.
2. Table: params, whole-model GFLOPs, throughput, median epoch time, final top-1, per-loop readout, separation.
3. Plots: results/acc_per_loop.png, results/separation_per_loop.png.

Run from the repo root:  PYTHONPATH=. .venv/bin/python tasks/T28_frozen_patch_cls_loop/plot.py
"""
import json
import logging
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

TASK_DIR = Path(__file__).resolve().parent
RESULTS = TASK_DIR / "results"
sys.path.insert(0, str(TASK_DIR))

from compare import (T11D, RunFiles, markdown_row, per_loop_readout, plot_curves,  # noqa: E402
                     separation_per_loop, train_summary)

from src.models import build  # noqa: E402

T28A = RunFiles("T28-a CLS-only loop", RESULTS / "unroll_T28-a.json", RESULTS / "exit_analysis.json",
                "T28-a_cls_only_loop", RESULTS / "cls_svd_T28-a", "tab:blue")
TOKENS, DIM, MLP_RATIO = 197, 192, 4

log = logging.getLogger("t28_plot")


@torch.no_grad()
def counted_flops(block: nn.Module) -> int:
    """FLOPs of one call of `block` on a (1, 197, 192) input (matmuls; 1 multiply-add = 2 FLOPs)."""
    block.eval()
    with FlopCounterMode(display=False) as counter:
        block(torch.randn(1, TOKENS, DIM))
    return counter.get_total_flops()


def formula_flops(n_tokens: int = TOKENS, dim: int = DIM) -> dict[str, int]:
    """Matmul FLOPs of one loop by formula: full block, CLS-only as coded, CLS-only with patch K/V cached."""
    hidden = MLP_RATIO * dim
    full = (2 * n_tokens * dim * 3 * dim + 4 * n_tokens * n_tokens * dim + 2 * n_tokens * dim * dim
            + 4 * n_tokens * dim * hidden)
    cls_cached = 2 * dim * dim + 4 * n_tokens * dim + 2 * dim * dim + 4 * dim * hidden  # q, scores + mix, proj, MLP
    cls_coded = cls_cached + 2 * n_tokens * dim * 2 * dim  # + keys and values of all tokens
    return {"full_loop": full, "cls_only_as_coded": cls_coded, "cls_only_kv_cached": cls_cached}


def attention_matmul_flops(n_queries: int, n_tokens: int = TOKENS, dim: int = DIM) -> int:
    """FLOPs of the attention scores q k^T and the mix A v for `n_queries` query rows."""
    return 4 * n_queries * n_tokens * dim


def flops_table() -> dict:
    """Counted and formula FLOPs of one loop, and the CLS-only share of a full loop.

    On CPU FlopCounterMode does not count PyTorch's fused attention kernel (scaled_dot_product_attention), so the
    counted numbers lack q k^T and A v; they are added back (`counted_with_attention`) and must match the formula.
    """
    full_block = list(build("unroll", unique_blocks=3, pretrained=False).blocks)[2]
    cls_block = list(build("unroll", unique_blocks=3, pretrained=False, cls_only_loop=True).blocks)[2]
    counted = {"full_loop": counted_flops(full_block), "cls_only_as_coded": counted_flops(cls_block)}
    counted_with_attention = {"full_loop": counted["full_loop"] + attention_matmul_flops(TOKENS),
                              "cls_only_as_coded": counted["cls_only_as_coded"] + attention_matmul_flops(1)}
    formula = formula_flops()
    for name, value in counted_with_attention.items():
        if value != formula[name]:
            raise RuntimeError(f"counted FLOPs {value} for {name} do not match the formula {formula[name]}")
    shares = {name: round(100.0 * value / formula["full_loop"], 2) for name, value in formula.items()}
    return {"counted_without_fused_attention": counted, "counted_with_attention": counted_with_attention,
            "formula": formula, "share_of_full_loop_pct": shares}


def main() -> None:
    """FLOPs, tables, plots."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    flops = flops_table()
    (RESULTS / "flops.json").write_text(json.dumps(flops, indent=1))
    runs = (T11D, T28A)
    summaries = {run.label: train_summary(run) for run in runs}
    readout = {run.label: per_loop_readout(run) for run in runs}
    separation = {(run.label, split): separation_per_loop(run, split) for run in runs for split in ("test", "train")}
    log.info("FLOPs of one loop: %s", flops)
    log.info(markdown_row(["model", "params", "GFLOPs / image", "throughput img/s", "median epoch s", "final top-1",
                           "best top-1", "sep. loop 10 (test / train)"]))
    for run in runs:
        s = summaries[run.label]
        log.info(markdown_row([run.label, f"{s['params']:,}", s["gflops"], s["throughput_img_s"], s["median_epoch_s"],
                               s["final_top1"], s["best_top1"],
                               f"{separation[(run.label, 'test')][-1]} / {separation[(run.label, 'train')][-1]}"]))
    for run in runs:
        log.info(markdown_row([f"{run.label} readout", *readout[run.label]]))
        log.info(markdown_row([f"{run.label} sep. test", *separation[(run.label, "test")]]))
    plot_curves({run.label: (readout[run.label], run.colour) for run in runs}, "test top-1 (%)",
                "T28 · readout after every loop", RESULTS / "acc_per_loop.png")
    plot_curves({f"{run.label} {split}": (separation[(run.label, split)], run.colour) for run in runs
                 for split in ("test",)}, "within / between class spread (lower = tighter)",
                "T28 · class separation after every loop (test)", RESULTS / "separation_per_loop.png")
    (RESULTS / "summary.json").write_text(json.dumps(
        {"flops": flops, "train": summaries, "readout": readout,
         "separation": {f"{label} {split}": v for (label, split), v in separation.items()}}, indent=1))


if __name__ == "__main__":
    main()
