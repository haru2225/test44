#!/usr/bin/env python3
"""Single-phase, periodic SiO2 crystal score generation. Positions and sigma: Angstrom.

test44 started as a narrowed ablation of test40 (same architecture, phase
dimension removed, trained on ONLY the crystal reference: 62 NPT thermal
snapshots of one fixed beta-cristobalite lattice, same topology throughout
-- the "easiest" possible case for a purely-relative-geometry model, no
glass-style network diversity or shared-model phase-conditioning to
confound the diagnosis).

The Score network now follows the EGNN implementation and score-network
wrapper from mila-iqia/diffusion_for_multi_scale_molecular_dynamics
(see egnn_vendor.py). Improved denoising is not established.
The periodic uplift preserves global translation invariance after projection
and does not supply an absolute lattice-origin reference.

This is a structure generator, not an energy/force model or an equilibrium
MD sampler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import ase.io
from ase import Atoms
from ase.data import chemical_symbols
from ase.neighborlist import primitive_neighbor_list
import numpy as np
import torch
from torch import nn

from egnn_vendor import EGNN

FORMAT = "test44-crystal-single-phase-v2"
CHECKPOINT_FORMAT = "test44-crystal-source-egnn-v1"
SI_MASS, O_MASS = 28.0855, 15.9994
BEAD_MASS = SI_MASS + 2 * O_MASS
DEFAULT_SIGMA_MAX = 1.5  # ~half the nearest bead-bead distance (2.96 A)
DEFAULT_EXTRA_NOISE = 2.0  # Angstrom, generation-time extra noise at step 0
DEFAULT_CUTOFF = 5.5
STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_pt(path, data):
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def load_pt(path):
    # Only load checkpoints produced locally or obtained from a trusted source.
    return torch.load(path, map_location="cpu", weights_only=False)


def output_dir(path, resume=False):
    if resume:
        if not path.is_dir():
            raise ValueError("Resume directory does not exist")
    elif path.exists() and any(path.iterdir()):
        raise ValueError(f"Output is not empty: {path}; use a new directory or --resume")
    path.mkdir(parents=True, exist_ok=True)
    return path


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def lengths_of(cell):
    cell = np.asarray(cell)
    lengths = np.diag(cell)
    if (not np.isfinite(cell).all() or np.any(lengths <= 0)
            or not np.allclose(cell, np.diag(lengths), atol=1e-6)):
        raise ValueError("Only positive, axis-aligned orthorhombic periodic cells are supported")
    return lengths


def tetrahedral_mapping(atoms, cutoff=2.2, allow_sharing_defects=False):
    """Shared-O mass-weighted tetrahedron centers; strict corner-sharing SiO2.

    Identical definition to test40's tetrahedral_mapping (same CG bead).
    """
    box = lengths_of(atoms.cell)
    if cutoff >= min(box) / 2:
        raise ValueError("Si-O mapping cutoff exceeds the minimum-image limit")
    z = atoms.numbers
    i, j, d = primitive_neighbor_list("ijD", pbc=atoms.pbc, cell=atoms.cell,
                                     positions=atoms.positions, cutoff=cutoff)
    mask = z[i] != z[j]
    degree = np.bincount(i[mask], minlength=len(atoms))
    valid_oxygen = np.all(degree[z == 8] >= 1) if allow_sharing_defects else np.all(degree[z == 8] == 2)
    if not np.all(degree[z == 14] == 4) or not valid_oxygen:
        bad_si = int(np.sum(degree[z == 14] != 4))
        bad_o = int(np.sum(degree[z == 8] != 2))
        raise ValueError(f"SiO4 mapping requires four O per Si and two Si per O; invalid Si={bad_si}, O={bad_o}. Check the reference/cutoff; defects are not silently remapped.")
    centers, sites = [], []
    for si in np.flatnonzero(z == 14):
        edges = np.flatnonzero(mask & (i == si))
        weights = O_MASS / degree[j[edges]]
        mass = SI_MASS + weights.sum()
        centers.append((atoms.positions[si] + (weights[:, None] * d[edges]).sum(0) / mass) % box)
        oxy = sorted(int(x) for x in j[edges])
        sites.append(dict(si_index=int(si), oxygen_indices=oxy,
                          oxygen_sharing_counts=[int(degree[x]) for x in oxy], mass_amu=float(mass)))
    return np.asarray(centers, dtype=np.float32), sites


def prepare(args):
    out = output_dir(args.output)
    frames, cell_lengths, seen, sources = [], [], set(), []
    numbers, mapping = None, None
    for path in args.input:
        sources.append(dict(name=path.name, sha256=digest(path)))
        options = {}
        if args.lammps_types:
            if args.input_format == "lammps-data":
                options["Z_of_type"] = dict(enumerate(args.lammps_types, 1))
            elif args.input_format == "lammps-dump-text":
                options["specorder"] = [chemical_symbols[z] for z in args.lammps_types]
        result = ase.io.read(path, index=args.index, format=args.input_format, **options)
        for atoms in result if isinstance(result, list) else [result]:
            atoms = atoms.repeat(args.repeat)
            if not atoms.pbc.all():
                raise ValueError(f"{path}: three periodic directions required")
            cell = lengths_of(atoms.cell)
            z = atoms.numbers
            if set(z) != {8, 14} or np.sum(z == 8) != 2 * np.sum(z == 14):
                raise ValueError(f"{path}: expected Si:O = 1:2 (atomic numbers 14 and 8)")
            if numbers is None:
                numbers = z.copy()
            if not np.array_equal(z, numbers):
                raise ValueError("All frames must have the same atom order")
            pos, sites = tetrahedral_mapping(atoms, args.mapping_cutoff, args.allow_sharing_defects)
            if mapping is not None and sites != mapping and not args.allow_topology_drift:
                raise ValueError("Si-O topology changed between frames; this fixed-topology mapping cannot represent bond exchange (pass --allow-topology-drift if that is intentional)")
            mapping = sites
            key = hashlib.sha256(pos.tobytes()).hexdigest()
            if key not in seen:
                frames.append(pos)
                cell_lengths.append(cell)
                seen.add(key)
    if not frames:
        raise ValueError("No frames")
    if len(frames) == 1:
        if not args.allow_single_reference:
            raise ValueError("One unique frame; provide independent frames or explicitly use --allow-single-reference")
        train_ids = valid_ids = [0]
        mode = "single-reference reconstruction; validation is NOT independent"
    else:
        nvalid = max(1, math.ceil(len(frames) * args.validation_fraction))
        split = len(frames) - nvalid
        train_ids, valid_ids = list(range(split - args.split_gap)), list(range(split, len(frames)))
        if not train_ids:
            raise ValueError("Too few training frames for validation fraction / split gap")
        mode = "ordered held-out tail; correlation depends on supplied frame spacing"
    np.save(out / "frames.npy", np.stack(frames))
    metadata = dict(format=FORMAT, units="Angstrom", numbers=[14] * len(mapping),
        lengths=[c.tolist() for c in cell_lengths],
        train_ids=train_ids, valid_ids=valid_ids, validation_mode=mode,
        frames=len(frames), sha256=digest(out / "frames.npy"), sources=sources,
        mapping=mapping, bead_masses_amu=[s["mass_amu"] for s in mapping],
        allow_sharing_defects=args.allow_sharing_defects, mapping_cutoff_A=args.mapping_cutoff)
    abnormal = {o: n for s in mapping for o, n in zip(s["oxygen_indices"], s["oxygen_sharing_counts"]) if n != 2}
    metadata["nonbridging_or_overcoordinated_oxygen"] = abnormal
    save_json(out / "metadata.json", metadata)
    print(f"{len(frames)} unique frames, {len(mapping)} SiO4 beads; {mode}; O sharing-count!=2: {len(abnormal)}", flush=True)
    return 0


def load_dataset(path):
    meta = json.loads((path / "metadata.json").read_text())
    if meta["format"] not in (FORMAT, "test44-crystal-single-phase-v1",
                              "test42-crystal-single-phase-v1", "test42-crystal-single-phase-v2"):
        raise ValueError("Not a test44 dataset")
    file = path / "frames.npy"
    if digest(file) != meta["sha256"]:
        raise ValueError(f"Dataset changed: {file}")
    return np.load(file, mmap_mode="r"), meta


def graph(pos, lengths, cutoff=None):
    """Directed graph on minimum-image distances; fully connected when cutoff is None (source default)."""
    n_atoms = pos.shape[0]
    indices = torch.arange(n_atoms, device=pos.device)
    src = indices.repeat_interleave(n_atoms)
    dst = indices.repeat(n_atoms)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    box = pos.new_tensor(lengths)
    delta = pos[dst] - pos[src]
    delta = delta - torch.round(delta / box) * box
    if cutoff is not None:
        near = delta.norm(dim=-1) < cutoff
        src, dst, delta = src[near], dst[near], delta[near]
    return src, dst, delta


def source_bloch_wave_vectors():
    """The three positive reciprocal vectors in the source's first cubic shell."""
    return torch.eye(3, dtype=torch.float32)


class Score(nn.Module):
    """Source EGNNScoreNetwork wrapper adapted to one SiO4 bead species.

    Matches the source's first cubic Bloch shell, sigma + atom-type one-hot
    features, fully connected minimum-image graph, Gamma projection, and
    configuration-template EGNN widths/depth. Output uses the source's
    sigma-normalized score convention.
    """
    def __init__(self, width=256, layers=4, hidden_layers=4, cutoff=DEFAULT_CUTOFF, bloch_shells=1):
        super().__init__()
        self.cutoff = cutoff  # Angstrom, minimum-image graph cutoff; None = fully connected
        if bloch_shells != 1:
            raise ValueError("Source-compatible EGNN uses the first cubic Bloch shell")
        self.spatial_dimension = 3
        self.num_atom_types = 1
        self.register_buffer("bloch_vectors", source_bloch_wave_vectors())
        generator = torch.tensor([[0., -1.], [1., 0.]])
        self.register_buffer("gamma", torch.stack([
            torch.block_diag(*[k[axis] * generator for k in self.bloch_vectors])
            for axis in range(3)
        ]))
        # one real species + MASK class + diffusion sigma
        self.egnn = EGNN(
            input_size=self.num_atom_types + 2,
            num_classes=self.num_atom_types + 1,
            message_n_hidden_dimensions=hidden_layers,
            message_hidden_dimensions_size=width,
            node_n_hidden_dimensions=hidden_layers,
            node_hidden_dimensions_size=width,
            coordinate_n_hidden_dimensions=hidden_layers,
            coordinate_hidden_dimensions_size=width,
            residual=True, attention=False, normalize=False, tanh=False,
            coords_agg="mean", message_agg="mean", n_layers=layers,
        )

    def uplift(self, fractional):
        kr = torch.einsum("kd,nd->nk", self.bloch_vectors.to(fractional),
                          2.0 * math.pi * fractional)
        return torch.stack([kr.cos(), kr.sin()], dim=-1).flatten(start_dim=1)

    def forward(self, types, pos, edges, sigma, lengths):
        src, dst, displacement = edges
        fractional = (pos / pos.new_tensor(lengths)) % 1.0
        z = self.uplift(fractional)
        sigma_features = pos.new_full((len(types), 1), math.log(sigma))  # log sigma feature (test44)
        # test44 has one unmasked bead species; source layout is [sigma, one-hot(type+MASK)].
        atom_type_features = pos.new_zeros((len(types), self.num_atom_types + 1))
        atom_type_features[:, 0] = 1.0
        h = torch.cat([sigma_features, atom_type_features], dim=-1)
        edge_features = torch.stack([src.to(pos.dtype), dst.to(pos.dtype),
                                     displacement.norm(dim=-1)], dim=-1)
        raw = self.egnn(h, edge_features, z.clone())
        # Source wrapper's projection returns sigma-normalized Cartesian scores;
        # no division by cell lengths is applied.
        return torch.einsum("ni,aij,nj->na", z, self.gamma, raw.X)

def wrapped_target(noisy, clean, lengths, sigma):
    """Exact periodic Gaussian sigma*score, with convergent image/Fourier sums."""
    box = torch.as_tensor(lengths, dtype=noisy.dtype, device=noisy.device)
    delta = (noisy - clean + box / 2) % box - box / 2
    result = torch.empty_like(delta)
    for axis in range(3):
        side = box[axis]
        d = delta[:, axis:axis + 1]
        if sigma / float(side) < 0.2:
            images = d + torch.arange(-2, 3, device=noisy.device) * side
            weights = torch.softmax(-0.5 * (images / sigma).square(), -1)
            result[:, axis] = -(weights * images).sum(-1) / sigma
        else:
            k = torch.arange(1, 13, dtype=noisy.dtype, device=noisy.device)
            amplitude = torch.exp(-2 * math.pi**2 * k.square() * (sigma / side)**2)
            angle = 2 * math.pi * d * k / side
            density = 1 + 2 * (amplitude * torch.cos(angle)).sum(-1)
            deriv = -(4 * math.pi / side) * (k * amplitude * torch.sin(angle)).sum(-1)
            result[:, axis] = sigma * deriv / density.clamp_min(1e-12)
    return result


def extra_noise_levels(steps, maximum, final_fraction):
    """Linear decay maximum -> 0 over the first (1 - final_fraction) of the steps, then exactly 0."""
    active = int(round(steps * (1 - final_fraction)))
    levels = np.zeros(steps)
    if active > 0:
        levels[:active] = maximum * (1 - np.arange(active) / active)
    return levels


def deadline(args):
    return time.monotonic() + args.time_budget_hours * 3600 if args.time_budget_hours else math.inf


def device_for(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu for local checks")
    return torch.device(name)


def train(args):
    arrays, meta = load_dataset(args.dataset)
    device = device_for(args.device)
    config = dict(width=args.width, layers=args.layers, hidden_layers=args.hidden_layers,
                  cutoff=args.cutoff or None)
    maximum = max(max(cell) for cell in meta["lengths"])
    # test44: sigma-max defaults to DEFAULT_SIGMA_MAX (Angstrom), not the cell size, so training
    # does not spend most steps on the near-uniform regime. The terminal distribution is therefore
    # NOT uniform; generation still starts from uniform positions (as the reference implementation does).
    sigma_max = args.sigma_max or DEFAULT_SIGMA_MAX
    if sigma_max <= args.sigma_min:
        raise ValueError("sigma-max must exceed sigma-min")
    settings = dict(dataset_sha256=digest(args.dataset / "metadata.json"), architecture=config,
        sigma_min=args.sigma_min, sigma_max=sigma_max, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size, seed=args.seed, device=args.device)
    output = output_dir(args.output, args.resume)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = Score(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    completed, history = 0, []
    if args.resume:
        ck = load_pt(output / "checkpoint.pt")
        if ck.get("format") != CHECKPOINT_FORMAT or ck["settings"] != settings:
            raise ValueError("Resume requires the same dataset and training settings")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
        completed, history = ck["step"], ck["history"]
    if args.updates < completed:
        raise ValueError("updates is below the completed checkpoint step")
    until = deadline(args)

    def save():
        save_pt(output / "checkpoint.pt", dict(format=CHECKPOINT_FORMAT, settings=settings, metadata=meta,
            model=model.state_dict(), optimizer=optimizer.state_dict(), step=completed,
            history=history, rng=rng_state()))
        save_json(output / "training.json", dict(settings=settings, step=completed, history=history))

    def loss_for(frame, sigma):
        lengths = meta["lengths"][frame]
        clean = torch.tensor(np.array(arrays[frame]), device=device)
        box = clean.new_tensor(lengths)
        noisy = (clean + sigma * torch.randn_like(clean)) % box
        types = torch.tensor(np.asarray(meta["numbers"]) == 14, device=device).long()
        pred = model(types, noisy, graph(noisy, lengths, model.cutoff), sigma, lengths)
        target = wrapped_target(noisy, clean, box, sigma)
        return (pred - target).square().mean(), target.square().mean()

    for step in range(completed + 1, args.updates + 1):
        if STOP or time.monotonic() >= until:
            save()
            return 75
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for _ in range(args.batch_size):
            frame = int(np.random.choice(meta["train_ids"]))
            sigma = math.exp(np.random.uniform(math.log(args.sigma_min), math.log(sigma_max)))
            loss, _ = loss_for(frame, sigma)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; last saved checkpoint is preserved")
            total = total + loss
        total = total / args.batch_size
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        completed = step
        train_loss = float(total.detach())
        if step == 1 or step % args.log_every == 0 or step == args.updates:
            state = rng_state()
            torch.manual_seed(args.seed + 10000)
            model.eval()
            metrics = {}
            with torch.no_grad():
                for label, sigma in (("small", args.sigma_min), ("local", min(0.3, sigma_max)),
                                     ("middle", math.sqrt(args.sigma_min * sigma_max)), ("terminal", sigma_max)):
                    values = [loss_for(frame, sigma) for frame in meta["valid_ids"][:8]]
                    metrics[label] = dict(mse=float(torch.stack([v[0] for v in values]).mean()),
                        zero_predictor_mse=float(torch.stack([v[1] for v in values]).mean()))
            restore_rng(state)
            row = dict(step=step, train=train_loss, validation=metrics)
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.checkpoint_every == 0:
            save()
    save()
    return 0


@torch.no_grad()
def generate(args):
    ck = load_pt(args.checkpoint)
    if ck.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("A source-compatible test44 EGNN v1 checkpoint is required; start a new run for changed optimizer settings")
    device = device_for(args.device)
    config = ck["settings"]["architecture"]
    model = Score(**config).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    meta = ck["metadata"]
    cell_frame = args.frame if args.frame is not None else meta["train_ids"][0]
    lengths = meta["lengths"][cell_frame]
    box = torch.tensor(lengths, dtype=torch.float32, device=device)
    types = torch.tensor(np.asarray(meta["numbers"]) == 14, device=device).long()
    levels = np.geomspace(ck["settings"]["sigma_max"], ck["settings"]["sigma_min"], args.steps + 1)
    settings = dict(checkpoint_sha256=digest(args.checkpoint), steps=args.steps,
                    extra_noise=args.extra_noise, extra_noise_final_fraction=args.extra_noise_final_fraction,
                    seed=args.seed, device=args.device, init="uniform_periodic", cell_frame=cell_frame)
    out = output_dir(args.output, args.resume)
    completed = 0
    if args.resume:
        state = load_pt(out / "restart.pt")
        if state["settings"] != settings:
            raise ValueError("Generation settings or checkpoint changed")
        pos, completed = state["positions"].to(device), state["step"]
        restore_rng(state["rng"])
        trajectory = np.lib.format.open_memmap(out / "positions.npy", mode="r+")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        pos = torch.rand((len(types), 3), device=device) * box
        trajectory = np.lib.format.open_memmap(out / "positions.npy", mode="w+", dtype="float32",
                                              shape=(args.steps + 1, len(types), 3))
        trajectory[0] = pos.cpu().numpy()
    until = deadline(args)

    def save():
        trajectory.flush()
        save_pt(out / "restart.pt", dict(settings=settings, positions=pos.cpu(), step=completed, rng=rng_state()))
        save_json(out / "generation.json", dict(format=FORMAT, settings=settings, step=completed,
            valid_frames=completed + 1, complete=completed == args.steps, numbers=meta["numbers"],
            lengths=lengths, is_equilibrium_trajectory=False))

    extra = extra_noise_levels(args.steps, args.extra_noise, args.extra_noise_final_fraction)

    for step in range(completed, args.steps):
        if STOP or time.monotonic() >= until:
            save()
            return 75
        sigma = float(levels[step])
        dv = float(levels[step]**2 - levels[step + 1]**2)
        pred = model(types, pos, graph(pos, lengths, model.cutoff), sigma, lengths)
        pos = (pos + dv / sigma * pred + math.sqrt(dv) * torch.randn_like(pos)
               + float(extra[step]) * torch.randn_like(pos)) % box
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite sample")
        completed = step + 1
        trajectory[completed] = pos.cpu().numpy()
        if completed % args.checkpoint_every == 0:
            save()
            print(f"{completed}/{args.steps}, sigma={sigma:.4g}", flush=True)
    save()
    atoms = Atoms(numbers=meta["numbers"], positions=pos.cpu().numpy(), cell=np.diag(lengths), pbc=True,
                  masses=meta["bead_masses_amu"])
    atoms.info.update(sigma_min_A=ck["settings"]["sigma_min"], is_equilibrium_sample=False,
                      representation="tetrahedron")
    ase.io.write(out / "final.extxyz", atoms)
    masses = atoms.get_masses()
    unique, ids = np.unique(np.round(masses, 8), return_inverse=True)
    with (out / "final.data").open("w") as stream:
        stream.write("test44 SiO4 CG positions; structure generator, no force field supplied\n\n")
        stream.write(f"{len(atoms)} atoms\n{len(unique)} atom types\n\n")
        for side, axis in zip(lengths, "xyz"):
            stream.write(f"0 {side:.12g} {axis}lo {axis}hi\n")
        stream.write("\nMasses\n\n")
        for k, mass in enumerate(unique, 1):
            stream.write(f"{k} {mass:.8f}\n")
        stream.write("\nAtoms # atomic\n\n")
        for k, (typ, xyz) in enumerate(zip(ids, atoms.positions), 1):
            stream.write(f"{k} {typ + 1} " + " ".join(f"{x:.10f}" for x in xyz) + "\n")
    return 0


def cg_stats(atoms, cutoff=4.0):
    """Bead-level network metrics; no invented internal Si-O distances/angles."""
    box = lengths_of(atoms.cell)
    radius = min(8.0, min(box) / 2 - 1e-6)
    if cutoff >= radius:
        raise ValueError("CG coordination cutoff must be smaller than the RDF radius / half-box")
    i, j, d = primitive_neighbor_list("ijD", pbc=atoms.pbc, cell=atoms.cell,
                                     positions=atoms.positions, cutoff=radius)
    r = np.linalg.norm(d, axis=1)
    degree = np.bincount(i[r < cutoff], minlength=len(atoms))
    bins = np.linspace(0, radius, 161)
    shell = 4 * np.pi / 3 * np.diff(bins**3)
    counts = np.histogram(r, bins)[0]
    rdf = counts * np.prod(box) / (len(atoms) * (len(atoms) - 1) * shell)
    grid = np.array([(a, b, c) for a in range(-4, 5) for b in range(-4, 5)
                     for c in range(-4, 5) if (a, b, c) != (0, 0, 0)])
    sq = np.abs(np.exp(2j * np.pi * ((atoms.positions / box) @ grid.T)).sum(0))**2 / len(atoms)
    return dict(beads=len(atoms), four_neighbor_fraction=float(np.mean(degree == 4)),
        coordination_hist=np.bincount(degree).tolist(), coordination_cutoff_A=cutoff,
        min_bead_distance_A=float(r.min()) if len(r) else None,
        rdf_r_A=((bins[:-1] + bins[1:]) / 2).tolist(), rdf=rdf.tolist(),
        reciprocal_indices=grid.tolist(), structure_factor=sq.tolist(),
        max_structure_factor=float(sq.max()))


def evaluate(args):
    arrays, meta = load_dataset(args.dataset)
    sample = ase.io.read(args.sample)
    all_lengths = np.asarray(meta["lengths"])
    sample_lengths = lengths_of(sample.cell)
    in_range = np.all((sample_lengths >= all_lengths.min(0) - 1e-5) & (sample_lengths <= all_lengths.max(0) + 1e-5))
    if not sample.pbc.all() or sorted(sample.numbers) != sorted(meta["numbers"]) or not in_range:
        raise ValueError("Sample composition/cell must match the dataset")
    cutoff = args.bond_cutoff or 4.0
    refs = [cg_stats(Atoms(numbers=meta["numbers"], positions=arrays[i],
        cell=np.diag(meta["lengths"][i]), pbc=True), cutoff) for i in meta["valid_ids"][:args.reference_frames]]
    report = dict(validation_mode=meta["validation_mode"], sample=cg_stats(sample, cutoff), references=refs,
                  note="No automatic scientific pass/fail. Compare coordination, RDF and reciprocal peaks across independent samples.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    print(json.dumps({k: report["sample"][k] for k in ("four_neighbor_fraction", "min_bead_distance_A", "max_structure_factor")}), flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ref in refs:
        axes[0].plot(ref["rdf_r_A"], ref["rdf"], color="C0", alpha=.4)
    axes[0].plot(report["sample"]["rdf_r_A"], report["sample"]["rdf"], color="C1", label="generated")
    axes[0].plot([], [], color="C0", label="reference")
    axes[0].set(xlabel="Bead distance (Angstrom)", ylabel="g(r)")
    axes[0].legend()
    wave = 2 * np.pi * np.asarray(refs[0]["reciprocal_indices"]) / sample_lengths
    magnitude = np.linalg.norm(wave, axis=1)
    axes[1].scatter(magnitude, np.mean([r["structure_factor"] for r in refs], axis=0), s=8, alpha=.4, label="reference")
    axes[1].scatter(magnitude, report["sample"]["structure_factor"], s=8, alpha=.4, label="generated")
    axes[1].set(xlabel="|k| (1/Angstrom)", ylabel="S(k), discrete reciprocal vectors")
    axes[1].legend()
    fig.suptitle("test44 SiO4 beads: crystal (single-phase ablation)")
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)
    return 0


def positive(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return value


def nonnegative(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("Must be finite and nonnegative")
    return value


def count(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--input", type=Path, nargs="+", required=True)
    prep.add_argument("--repeat", type=count, nargs=3, default=(1, 1, 1))
    prep.add_argument("--input-format", default=None, help="ASE format, e.g. lammps-data or lammps-dump-text; auto-detected if omitted")
    prep.add_argument("--lammps-types", type=int, nargs="+", help="Atomic numbers for LAMMPS types 1,2,...")
    prep.add_argument("--index", default=":", help="ASE frame slice, e.g. ::10")
    prep.add_argument("--validation-fraction", type=positive, default=0.1)
    prep.add_argument("--split-gap", type=int, default=0, help="Discard this many frames before held-out tail")
    prep.add_argument("--allow-single-reference", action="store_true")
    prep.add_argument("--mapping-cutoff", type=positive, default=2.2, help="Si-O distance defining shared SiO4 tetrahedra, Angstrom")
    prep.add_argument("--allow-sharing-defects", action="store_true", help="Allow O shared by 1 or >2 tetrahedra; divide its mass by actual sharing count. Si still must have four O.")
    prep.add_argument("--allow-topology-drift", action="store_true", help="Allow frames whose Si-O connectivity differs. Training is unaffected; only the single stored bead_masses_amu reflects one frame.")
    prep.add_argument("--output", type=Path, required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", type=Path, required=True)
    tr.add_argument("--updates", type=count, default=30000)
    tr.add_argument("--batch-size", type=count, default=8, help="(frame, sigma) samples averaged before each backward")
    tr.add_argument("--width", type=count, default=256)
    tr.add_argument("--layers", type=count, default=4)
    tr.add_argument("--hidden-layers", type=count, default=4)
    tr.add_argument("--cutoff", type=nonnegative, default=DEFAULT_CUTOFF, help="Graph cutoff, Angstrom (minimum-image); 0 = fully connected")
    tr.add_argument("--sigma-min", type=positive, default=0.03)
    tr.add_argument("--sigma-max", type=positive)
    tr.add_argument("--learning-rate", type=positive, default=2e-4)
    tr.add_argument("--weight-decay", type=positive, default=5e-8)
    tr.add_argument("--log-every", type=count, default=100)
    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", type=Path, required=True)
    gen.add_argument("--steps", type=count, default=3000)
    gen.add_argument("--extra-noise", type=float, default=DEFAULT_EXTRA_NOISE,
                     help="Extra Gaussian noise (A) added each step, decaying linearly to 0; 0 disables (test43 behaviour)")
    gen.add_argument("--extra-noise-final-fraction", type=float, default=0.03,
                     help="Final fraction of steps run without extra noise")
    gen.add_argument("--frame", type=int, help="Dataset frame index to use as the fixed generation cell; default is the first training frame")
    for q in (tr, gen):
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        q.add_argument("--seed", type=int, default=1337)
        q.add_argument("--checkpoint-every", type=count, default=100)
        q.add_argument("--time-budget-hours", type=float, default=19.5, help="0 disables deadline")
        q.add_argument("--resume", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--dataset", type=Path, required=True)
    ev.add_argument("--sample", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--reference-frames", type=count, default=4)
    ev.add_argument("--bond-cutoff", type=positive, help="Coordination cutoff, default 4.0 A")
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare" and (args.validation_fraction >= 1 or args.split_gap < 0):
        raise ValueError("validation-fraction must be <1 and split-gap nonnegative")
    if hasattr(args, "time_budget_hours") and (not math.isfinite(args.time_budget_hours) or args.time_budget_hours < 0):
        raise ValueError("time-budget-hours must be finite and nonnegative")
    if args.command == "generate" and (args.extra_noise < 0 or not 0 <= args.extra_noise_final_fraction < 1):
        raise ValueError("extra-noise must be >= 0 and extra-noise-final-fraction in [0, 1)")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return globals()[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
