# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Iterative ViT-Tiny: a narrated walkthrough
#
# **The question.** Take a pretrained ViT-Tiny (a small image model with 12 "blocks" stacked on top of each other).
# Keep **one** block and apply that *same* block 12 times. We save 88% of the parameters. How much accuracy do we lose?
#
# We compare, on CIFAR-100 (100 classes of small photos), fine-tuned with an identical recipe:
# 1. **baseline**: the normal ViT-Tiny, 12 different blocks;
# 2. **iterative, block 0**: block 0 reused 12 times;
# 3. **iterative, block 6**: block 6 reused 12 times.
#
# **How to use this notebook.** Read the text above each cell, guess the answer where it says *Predict first*, then run the cell.
# Run top to bottom. "Restart and run all" works.
#
# **Where is the real code?** The repo's `src/` folder holds the script version of everything here
# (`src/models.py`, `src/data.py`, `src/metrics.py`, `src/train.py`). This notebook re-implements it in small cells
# so it is self-contained and does not import from `src/`.
#
# **Colab warnings.** Free Colab sessions can disconnect after idle time, and a GPU is not guaranteed.
# That is why every epoch of every run is saved to a JSON file as it finishes (and optionally copied to Google Drive).

# %% [markdown]
# ## 1. Setup and device check
#
# **What this cell does.** Two switches control everything:
# - `SMALL = True` is a *tiny mode* (256 train and 256 test images, 1 epoch). It runs on a laptop CPU in a few minutes. Its accuracy numbers are meaningless; it only proves the code works.
# - `SMALL = False` is the *real experiment* (all 50,000 train and 10,000 test images, 10 epochs). It needs a GPU.
#
# `DRIVE = False` means Google Drive is never touched. Set it to `True` only if you want results copied to your Drive.

# %%
SMALL = True    # True: tiny CPU-friendly test. False: full CIFAR-100, 10 epochs, needs a GPU.
DRIVE = False   # True: also copy result files to Google Drive (asks you to authorise the mount).
SEED = 0
print(f"SMALL={SMALL}  DRIVE={DRIVE}  SEED={SEED}")

# %% [markdown]
# **What this cell does.** Installs `timm` only if it is missing (Colab already has PyTorch), then prints versions
# and what hardware we got.
#
# New terms:
# - **tensor**: a grid of numbers (like a NumPy array) that PyTorch can run on a GPU and differentiate. An image batch is a tensor of shape `(batch, 3, 224, 224)`.
# - **device**: where tensors live and are computed, `cpu` or `cuda` (an NVIDIA GPU).
# - **AMP** (automatic mixed precision): do most maths in 16-bit numbers on the GPU. Faster and lighter, with nearly the same accuracy. We only turn it on when a GPU is present.

# %%
import importlib.util, subprocess, sys

if importlib.util.find_spec("timm") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)

import torch, timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
if not SMALL:
    assert torch.cuda.is_available(), (
        "Full mode needs a GPU. In Colab pick: Runtime > Change runtime type > T4 GPU, then rerun.")
print("torch", torch.__version__, "| timm", timm.__version__)
print("device:", device, "| GPU:", torch.cuda.get_device_name(0) if USE_AMP else "none (CPU)", "| AMP:", USE_AMP)

# %% [markdown]
# **What this cell does.** Turns the `SMALL` switch into concrete settings, and finds the dataset folder.
# `epochs` is how many times the model sees the whole training set. `batch_size` is how many images are processed at once.
# `num_workers` are helper processes that load images (0 is the safe choice on a laptop; Colab has only 2 CPU cores).

# %%
from pathlib import Path

if SMALL:
    CFG = dict(subset=256, epochs=1, batch_size=32, num_workers=0, tp_batch=32, tp_iters=5)
else:
    CFG = dict(subset=0, epochs=10, batch_size=128, num_workers=2, tp_batch=256, tp_iters=30)
# Reuse the repo's data folder if the notebook sits inside the repo; otherwise download into ./data (Colab).
DATA_ROOT = "../data" if Path("../data/cifar-100-python").exists() else "data"
OUT_DIR = Path("nb_results"); OUT_DIR.mkdir(exist_ok=True)
print(CFG); print("data root:", DATA_ROOT, "| results folder:", OUT_DIR)

# %% [markdown]
# ## 2. Load ViT-Tiny and look at its 12 blocks
#
# **What this cell does.** Downloads pretrained ViT-Tiny from `timm` (a library of image models) and replaces its
# final layer with a fresh 100-class one (CIFAR-100 has 100 classes).
#
# New terms:
# - **nn.Module**: PyTorch's building block for models. A model is a Module that contains other Modules. `print(model)` shows the tree.
# - **block**: one repeated unit of the transformer. It does two things to every image patch: *attention* (patches look at each other) and an *MLP* (each patch is processed on its own).
# - **residual connection**: each sub-step adds its result to its input (`x = x + f(x)`), so a block only has to learn a small *change*. This is why a block can be applied again and again without exploding.
# - **LayerNorm**: rescales each patch's numbers to a stable range before attention and MLP.

# %%
import torch.nn as nn

MODEL_NAME, NUM_CLASSES = "vit_tiny_patch16_224", 100

def build_baseline():
    """Normal ViT-Tiny: 12 different blocks, new 100-class head."""
    return timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)

model = build_baseline()
print("number of blocks:", len(model.blocks))
print(model.blocks[0])

# %% [markdown]
# ## 3. Parameters per block
#
# A **parameter** is one learned number (a weight). Counting them tells us how big the model is.
#
# **Predict first:** how many parameters does ONE block have? (Hint: the model width is 192, and the MLP expands it 4x to 768 and back.)

# %%
def n_params(module):
    return sum(p.numel() for p in module.parameters())

for i, b in enumerate(model.blocks):
    print(f"block {i:2d}: {n_params(b):,} params")
print(f"total model (12 blocks + patch embed + new head): {n_params(model):,}")

# %% [markdown]
# Every block has 444,864 parameters, 12 of them make 5.34 M, and the whole model is 5,543,716.
# So the blocks are almost all of the model. Sharing one block should remove about 11 x 444,864 = 4.9 M parameters.

# %% [markdown]
# ## 4. One forward pass and tensor shapes
#
# A **forward pass** feeds an image through the model to get predictions. ViT cuts the 224x224 image into 16x16 patches
# (14 x 14 = 196 patches), turns each into a vector of 192 numbers, adds one special "class" token (197 tokens in total), and
# runs them through the blocks.
#
# **Predict first:** what shape does a batch of 1 image have after the blocks? And what comes out at the end?

# %%
x = torch.randn(1, 3, 224, 224)           # one fake image: (batch, channels, height, width)
model.eval()
with torch.no_grad():                     # no_grad: we only look, so skip gradient bookkeeping
    h = model.patch_embed(x);  print("after patch_embed:", tuple(h.shape))
    h = model._pos_embed(h);   print("+ class token + position:", tuple(h.shape))
    for i, b in enumerate(model.blocks):
        h = b(h)
        if i in (0, 11): print(f"after block {i}:", tuple(h.shape))
    print("final output (class scores):", tuple(model(x).shape))

# %% [markdown]
# A block takes `(1, 197, 192)` and returns the same shape. **That is the key fact**: because input and output shapes match,
# the output of a block can be fed straight back into the same block.

# %% [markdown]
# ## 5. Build the iterative model and PROVE the weights are shared
#
# **What this cell does.** `[block] * 12` makes a list with the *same Python object* 12 times (not 12 copies).
# We put that into the model in place of the 12 different blocks. One set of weights, used 12 times.
# (`copy.deepcopy` would silently make 12 independent copies, which is the classic mistake.)

# %%
def build_iterative(block_idx=0):
    """ViT-Tiny whose 12 blocks are ONE pretrained block (block_idx) applied 12 times."""
    m = build_baseline()
    shared = m.blocks[block_idx]
    m.blocks = nn.Sequential(*[shared] * len(m.blocks))
    return m

it = build_iterative(0)
ids = [id(b) for b in it.blocks]
print("object ids of the 12 slots:", ids)
print("distinct objects:", len(set(ids)), "| blocks[0] is blocks[11]:", it.blocks[0] is it.blocks[11])

# %% [markdown]
# **Predict first:** how many parameters does the iterative model have? (Baseline minus eleven blocks.)

# %%
base_n, it_n = n_params(model), n_params(it)
print(f"baseline: {base_n:,} | iterative: {it_n:,} | saved: {base_n - it_n:,}")
assert it_n == 650_212 and base_n - it_n == 11 * n_params(model.blocks[0])
print("650,212 = 5,543,716 - 11 x 444,864  ->  confirmed")

# %% [markdown]
# **Do gradients reach the shared block?** A **gradient** says how to nudge each weight to reduce the error;
# `backward()` computes them. If sharing works, the single block collects gradient from all 12 uses.

# %%
it.train(); it.zero_grad()
it(torch.randn(2, 3, 224, 224)).sum().backward()
grads = [p.grad is not None and p.grad.abs().sum().item() > 0 for p in it.blocks[0].parameters()]
print(f"{sum(grads)} of {len(grads)} parameter tensors in the shared block received a non-zero gradient")
assert all(grads)

# %% [markdown]
# ## 6. CIFAR-100 with timm, using the model's own preprocessing
#
# Pretrained models expect images prepared a specific way: size (224 x 224) and colour normalisation (subtract a mean, divide by a std).
# `resolve_data_config` reads exactly what ViT-Tiny was trained with, and `create_transform` builds the pipeline.
# CIFAR images are only 32x32, so they get enlarged to 224. The training transform also adds light random augmentation; the test transform does not.

# %%
from timm.data import create_dataset, create_transform, resolve_data_config
from torch.utils.data import DataLoader, Subset

data_cfg = resolve_data_config({}, model=model)
print({k: data_cfg[k] for k in ("input_size", "mean", "std", "interpolation")})

def make_loaders(m, subset, batch_size, num_workers):
    """Train and test DataLoaders. subset=0 means use everything."""
    cfg, loaders = resolve_data_config({}, model=m), []
    for split, training in (("train", True), ("validation", False)):
        ds = create_dataset("torch/cifar100", root=DATA_ROOT, split=split, download=True)
        ds.transform = create_transform(**cfg, is_training=training)
        if subset:
            g = torch.Generator().manual_seed(SEED)
            ds = Subset(ds, torch.randperm(len(ds), generator=g)[:subset].tolist())
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=training, num_workers=num_workers,
                                  pin_memory=USE_AMP, drop_last=training))
    return loaders

train_loader, test_loader = make_loaders(model, **{k: CFG[k] for k in ("subset", "batch_size", "num_workers")})
print("train images:", len(train_loader.dataset), "| test images:", len(test_loader.dataset),
      "| train batches per epoch:", len(train_loader))

# %% [markdown]
# **What this cell does.** Shows 8 training images (after the transform, so slightly augmented and blurry from enlarging),
# undoing the colour normalisation just for display.

# %%
import matplotlib.pyplot as plt

xb, yb = next(iter(train_loader))
mean = torch.tensor(data_cfg["mean"]).view(3, 1, 1); std = torch.tensor(data_cfg["std"]).view(3, 1, 1)
base_ds = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
names = getattr(base_ds, "classes", None)
fig, axes = plt.subplots(2, 4, figsize=(8, 4.4))
for ax, img, lab in zip(axes.flat, xb, yb):
    ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0)); ax.axis("off")
    ax.set_title(names[lab] if names else int(lab), fontsize=9)
plt.tight_layout(); plt.show()
print("one batch:", tuple(xb.shape), "labels:", tuple(yb.shape))

# %% [markdown]
# ## 7. The training loop, explained line by line
#
# Training = repeat: *predict -> measure error -> compute gradients -> nudge weights*. New terms:
# - **loss**: one number for how wrong the predictions are. We use cross-entropy with **label smoothing** 0.1 (targets are 90% "correct class" plus a little spread over the others, which stops over-confidence).
# - **optimizer**: the rule that nudges weights using gradients. We use **AdamW** with learning rate 1e-4 (step size) and weight decay 0.05 (a gentle pull of weights toward zero).
# - **epoch**: one full pass over the training set.
# - **learning-rate schedule**: the step size starts near 0, ramps up during 1 *warmup* epoch, then decays smoothly along a cosine curve to 0.
#
# First: seeding (so runs are repeatable) and the schedule, with a plot.

# %%
import math, random
import numpy as np

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:                                    # warmup: ramp up linearly
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)  # 0 -> 1 over the rest of training
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

set_seed(SEED)
curve = [cosine_lr(s, 1000, 100, 1e-4) for s in range(1000)]
plt.figure(figsize=(5, 2.6)); plt.plot(curve); plt.xlabel("step"); plt.ylabel("learning rate"); plt.show()
print(f"step 0: {curve[0]:.2e} | peak: {max(curve):.2e} | last: {curve[-1]:.2e}")

# %% [markdown]
# **Evaluation.** `model.eval()` switches off training-only behaviour (like augmentation-style randomness inside layers),
# and `torch.no_grad()` skips gradient bookkeeping. Accuracy = percent of test images where the highest-scoring class is the true one.

# %%
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=USE_AMP):
            pred = model(x).argmax(1)             # index of the highest class score
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total

print("accuracy of the untrained baseline head on the test set (chance is 1%):",
      f"{evaluate(model.to(device), test_loader):.2f}%")

# %% [markdown]
# **One epoch of training**, with a comment on every line. The `GradScaler` is part of AMP: 16-bit gradients can underflow to 0,
# so the loss is multiplied by a large number first and divided back before the update. It is switched off on CPU.

# %%
def train_one_epoch(model, loader, opt, criterion, scaler, step, total_steps, warmup_steps, base_lr):
    model.train()                                         # training mode
    loss_sum = 0.0
    for x, y in loader:
        for g in opt.param_groups:                        # set this step's learning rate
            g["lr"] = cosine_lr(step, total_steps, warmup_steps, base_lr)
        x, y = x.to(device), y.to(device)                 # move the batch to the GPU/CPU
        with torch.autocast(device.type, enabled=USE_AMP):
            loss = criterion(model(x), y)                 # 1. predict, 2. measure error
        opt.zero_grad(set_to_none=True)                   # forget old gradients
        scaler.scale(loss).backward()                     # 3. compute gradients
        scaler.step(opt)                                  # 4. nudge the weights
        scaler.update()
        loss_sum += loss.item(); step += 1
    return loss_sum / len(loader), step

print("defined train_one_epoch")

# %% [markdown]
# **Metrics.** Three numbers describe a model besides accuracy:
# - **parameters**: how many learned numbers it stores (shared weights counted once, since `.parameters()` lists a shared block once);
# - **FLOPs**: how many arithmetic operations one image needs (1 multiply-add = 2 FLOPs). This measures *compute*;
# - **throughput**: images per second at inference. This measures *real speed* on this hardware.
#
# **Predict first:** will the iterative model need fewer FLOPs than the baseline? (It still runs 12 blocks.)

# %%
import time
from torch.utils.flop_counter import FlopCounterMode

@torch.no_grad()
def count_gflops(model):
    model.eval()
    with FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 3, 224, 224, device=device))
    return fc.get_total_flops() / 1e9

@torch.no_grad()
def measure_throughput(model, batch, iters, warmup=2):
    model.eval()
    x = torch.randn(batch, 3, 224, 224, device=device)
    sync = torch.cuda.synchronize if USE_AMP else (lambda: None)
    with torch.autocast(device.type, enabled=USE_AMP):
        for _ in range(warmup): model(x)
        sync(); t0 = time.perf_counter()
        for _ in range(iters): model(x)
        sync()
    return batch * iters / (time.perf_counter() - t0)

b, i_ = build_baseline().to(device), build_iterative(0).to(device)
print(f"GFLOPs  baseline {count_gflops(b):.2f} | iterative {count_gflops(i_):.2f}  (equal: same compute)")
del b, i_

# %% [markdown]
# **Putting it together.** `run_experiment` builds a model, measures its metrics, trains it for `epochs`, evaluates after each epoch,
# and **writes the result JSON after every epoch** (so a dropped Colab session still leaves partial results).
# The JSON has the same fields as the server runs in `docs/results/`.

# %%
import json, shutil

def run_experiment(name, builder, epochs, subset, batch_size, num_workers, lr=1e-4, wd=0.05, warmup_epochs=1.0):
    set_seed(SEED)
    model = builder().to(device)
    train_l, test_l = make_loaders(model, subset, batch_size, num_workers)
    info = {"name": name, "device": str(device), "params": n_params(model), "gflops": count_gflops(model),
            "throughput_img_s": measure_throughput(model, CFG["tp_batch"], CFG["tp_iters"]), "history": []}
    print(f"[{name}] params={info['params']:,} GFLOPs={info['gflops']:.2f} throughput={info['throughput_img_s']:.0f} img/s")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    total, warm, step, t_run = epochs * len(train_l), int(warmup_epochs * len(train_l)), 0, time.time()
    for ep in range(epochs):
        t0 = time.time()
        loss, step = train_one_epoch(model, train_l, opt, criterion, scaler, step, total, warm, lr)
        acc = evaluate(model, test_l)
        info["history"].append({"epoch": ep + 1, "train_loss": loss, "test_acc": acc, "epoch_time_s": time.time() - t0})
        info.update(final_test_acc=acc, best_test_acc=max(h["test_acc"] for h in info["history"]),
                    total_time_s=time.time() - t_run)
        save_result(name, info)
        print(f"[{name}] epoch {ep + 1}/{epochs} loss={loss:.4f} acc={acc:.2f} ({time.time() - t0:.0f}s)")
    print(f"[{name}] total time: {info['total_time_s']:.0f}s")
    return info

print("defined run_experiment")

# %% [markdown]
# **Saving results.** `save_result` writes `nb_results/<name>.json` on the Colab disk. If `DRIVE=True`, it also mounts Google Drive
# (Colab will ask you to authorise; nothing is mounted otherwise) and copies the file into `MyDrive/iterative_vit/`.

# %%
DRIVE_DIR = None
if DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/iterative_vit"); DRIVE_DIR.mkdir(parents=True, exist_ok=True)

def save_result(name, info):
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(info, indent=2))
    if DRIVE_DIR is not None:
        shutil.copy(path, DRIVE_DIR / path.name)

print("results go to:", OUT_DIR.resolve().name + "/", "| Drive copy:", DRIVE_DIR is not None)

# %% [markdown]
# ## 8. A tiny run (always tiny, on any hardware)
#
# Before the long runs, see the loop work end to end on 256 images for 1 epoch (iterative, block 0).
#
# **Predict first:** what accuracy do you expect after 1 epoch on 256 images? (Chance is 1%.) Expect something near chance: this only proves the loop runs.

# %%
tiny = run_experiment("tiny_check", lambda: build_iterative(0), epochs=1, subset=256, batch_size=32, num_workers=0)
print("tiny-run accuracy:", tiny["final_test_acc"], "% (near chance is normal)")

# %% [markdown]
# ## 9. The runs (baseline, block 0, block 6)
#
# The three models run one after another with the same recipe and the same seed. Each reseeds first so the comparison is fair.
# - With `SMALL = True` this is 1 tiny epoch each, a few minutes on a laptop.
# - With `SMALL = False` it is 10 full epochs each. On a Colab T4 that is roughly 15 to 25 minutes per model (a rough estimate, not measured; the A4000 server took about 8 minutes). The first epoch also downloads CIFAR-100 (about 160 MB).
#
# If Colab disconnects, the per-epoch JSON files in `nb_results/` (and Drive, if enabled) keep what finished.

# %%
RUNS = {"baseline": build_baseline,
        "iterative_b0": lambda: build_iterative(0),
        "iterative_b6": lambda: build_iterative(6)}
run_kwargs = {k: CFG[k] for k in ("epochs", "subset", "batch_size", "num_workers")}
results = {}
for name, builder in RUNS.items():
    results[name] = run_experiment(name, builder, **run_kwargs)
print("finished:", list(results))

# %% [markdown]
# ## 10. Results: table and accuracy plot
#
# **Table.** One row per model from this notebook run.

# %%
import pandas as pd

df = pd.DataFrame([{"model": k, "params (M)": round(v["params"] / 1e6, 3), "GFLOPs": round(v["gflops"], 2),
                    "throughput (img/s)": round(v["throughput_img_s"]), "final acc (%)": round(v["final_test_acc"], 2),
                    "best acc (%)": round(v["best_test_acc"], 2), "time (s)": round(v["total_time_s"])}
                   for k, v in results.items()])
print(df.to_string(index=False))

# %% [markdown]
# **Compare with the server results.** The repo is public, so we fetch the saved A4000 results from GitHub
# (`docs/results/*.json`). If you are offline, this cell just skips.

# %%
import urllib.request

RAW = "https://raw.githubusercontent.com/reddy-nithin/Iterative-Transformers/main/docs/results/"
server = {}
for ours, theirs in {"baseline": "baseline", "iterative_b0": "iterative", "iterative_b6": "iterative_b6"}.items():
    try:
        with urllib.request.urlopen(f"{RAW}{theirs}.json", timeout=10) as r:
            server[ours] = json.load(r)
    except Exception as e:
        print(f"skipping server results for {ours} ({type(e).__name__}); continuing without them")
for k, v in server.items():
    print(f"server A4000 {k}: final acc {v['final_test_acc']:.2f}%, params {v['params']:,}")

# %% [markdown]
# **Plot.** Test accuracy after each epoch: solid lines are this notebook, dashed lines the A4000 server runs (if downloaded).
# In `SMALL` mode there is only one epoch, so you will see single dots near chance, not curves.

# %%
fig, ax = plt.subplots(figsize=(6, 3.8))
for i, (k, v) in enumerate(results.items()):
    ax.plot([h["epoch"] for h in v["history"]], [h["test_acc"] for h in v["history"]],
            marker="o", color=f"C{i}", label=f"{k} (this run)")
    if k in server:
        ax.plot([h["epoch"] for h in server[k]["history"]], [h["test_acc"] for h in server[k]["history"]],
                "--", color=f"C{i}", alpha=0.6, label=f"{k} (A4000 server)")
ax.set_xlabel("epoch"); ax.set_ylabel("CIFAR-100 test accuracy (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Caution about speed numbers.** Throughput and FLOPs from Colab (or from your Mac) are **not comparable** with the A4000 numbers:
# different hardware gives different img/s, and CPU and GPU even count attention FLOPs differently (about 2.15 G on CPU vs 2.51 G on GPU).
# Compare models only against each other *within the same run*. Accuracy is the number that transfers across machines (up to small run-to-run noise).

# %% [markdown]
# ## 11. Interpretation and next ideas
#
# **Sanity check against the known A4000 results** (full mode, 10 epochs, seed 0):
#
# | Model | Params | Final test acc |
# |---|---|---|
# | Baseline | 5.544 M | 86.77% |
# | Iterative, block 0 x 12 | 0.650 M | 34.76% |
# | Iterative, block 6 x 12 | 0.650 M | 30.70% |
#
# A Colab T4 run should land within about a point or two of these (GPU kernels are not bit-identical across machines). If you ran `SMALL = True`, your accuracy is near chance and says nothing about the idea.
#
# **What it means.**
# - **Size vs accuracy.** Parameters drop 88% but accuracy collapses from about 87% to about 31 to 35%. Naively sharing one pretrained block does not work under this recipe.
# - **No compute saving.** FLOPs are identical because the block still runs 12 times. Sharing saves memory and storage, not time.
# - **Block choice.** A middle block (6) did *not* beat block 0. One seed and 10 epochs cannot rank them.
# - **Not converged.** Both iterative models were still improving at epoch 10, and lr 1e-4 was chosen for ordinary fine-tuning.
#
# **Next ideas.** (1) Try all 12 block indices. (2) Higher learning rate and a longer schedule for the shared model. (3) Repeat with 2 to 3 seeds.
# (4) A gentler design: share within groups of blocks, or add small per-iteration adapters.
