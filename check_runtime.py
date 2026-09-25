"""Check bundled data and one EGNN forward/backward pass on the GPU node."""
import argparse
from pathlib import Path

import torch

from test44 import Score, graph, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("examples/crystal"))
    args = parser.parse_args()
    frames, meta = load_dataset(args.dataset)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: run on a GPU node with singularity/apptainer exec --nv")
    torch.manual_seed(42)
    pos = torch.tensor(frames[0].copy(), device="cuda")
    box = meta["lengths"][0]
    model = Score(width=32, layers=4, hidden_layers=1).cuda()
    prediction = model(torch.ones(len(pos), device="cuda", dtype=torch.long),
                       pos, graph(pos, box, model.cutoff), .3, box)
    prediction.square().mean().backward()
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Non-finite GPU prediction")
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Missing or non-finite GPU gradients")
    print(f"GPU check passed: {torch.cuda.get_device_name()}, torch={torch.__version__}, "
          f"CUDA={torch.version.cuda}, frames={len(frames)}, beads={len(pos)}")


if __name__ == "__main__":
    main()
