"""Print ViT-Tiny structure and parameters per block (run: python -m scripts.inspect_model)."""
from src.models import build_baseline

m = build_baseline(pretrained=True)
print(f"blocks: {len(m.blocks)}")
print(m.blocks[0])
for i, b in enumerate(m.blocks):
    print(f"block {i:2d}: {sum(p.numel() for p in b.parameters()):,} params")
print(f"total (with new 100-class head): {sum(p.numel() for p in m.parameters()):,}")
