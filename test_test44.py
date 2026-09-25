"""CPU end-to-end and unit checks; not a quality benchmark. Adapted from
test40's suite, minus phase-conditioning (test44 is single-phase)."""
import json

import ase.io
from ase import Atoms
import numpy as np
import pytest
import torch

import test44 as t

torch.set_num_threads(1)


def corner_sharing_crystal():
    from ase.build import bulk
    from ase.neighborlist import primitive_neighbor_list
    si = bulk("Si", "diamond", a=7.2, cubic=True)
    i, j, d = primitive_neighbor_list("ijD", pbc=si.pbc, cell=si.cell, positions=si.positions, cutoff=3.2)
    oxygens = (si.positions[i[i < j]] + d[i < j] / 2) % 7.2
    atoms = Atoms(numbers=[14] * len(si) + [8] * len(oxygens),
                  positions=np.vstack((si.positions, oxygens)), cell=si.cell, pbc=True)
    return atoms.repeat((2, 2, 2))


def test_tetrahedron_mapping_mass_periodicity_and_defects():
    atoms = corner_sharing_crystal()
    centers, sites = t.tetrahedral_mapping(atoms)
    assert len(centers) == len(atoms) // 3
    diff = centers - atoms.positions[atoms.numbers == 14]
    diff -= 14.4 * np.round(diff / 14.4)
    np.testing.assert_allclose(diff, 0, atol=1e-6)
    assert len(centers) * t.BEAD_MASS == pytest.approx(
        np.sum(atoms.numbers == 14) * t.SI_MASS + np.sum(atoms.numbers == 8) * t.O_MASS)
    oxygen = int(np.flatnonzero(atoms.numbers == 8)[0])
    atoms.positions[oxygen, 0] += .1
    shifted, _ = t.tetrahedral_mapping(atoms)
    delta = shifted - centers
    delta -= 14.4 * np.round(delta / 14.4)
    affected = [i for i, site in enumerate(sites) if oxygen in site["oxygen_indices"]]
    assert len(affected) == 2
    np.testing.assert_allclose(delta[affected, 0], .1 * t.O_MASS / 2 / t.BEAD_MASS, atol=1e-6)
    atoms.positions[oxygen] += [3., 2., 1.]
    with pytest.raises(ValueError, match="mapping requires"):
        t.tetrahedral_mapping(atoms)


def dataset(tmp_path, n_frames=3):
    crystal = corner_sharing_crystal()
    rng = np.random.default_rng(9)
    frames = []
    for _ in range(n_frames):
        a = crystal.copy()
        a.positions += rng.normal(0, .02, a.positions.shape)
        frames.append(a)
    path = tmp_path / "crystal.extxyz"
    ase.io.write(path, frames)
    args = t.parser().parse_args(["prepare", "--input", str(path), "--output", str(tmp_path / "data")])
    t.prepare(args)
    return tmp_path / "data"


def test_end_to_end_prepare_train_generate_evaluate(tmp_path):
    data = dataset(tmp_path)
    arrays, meta = t.load_dataset(data)
    assert meta["frames"] == 3
    assert arrays.shape == (3, 64, 3)

    out = tmp_path / "train"
    t.train(t.parser().parse_args(["train", "--dataset", str(data), "--output", str(out), "--device", "cpu",
                                   "--width", "8", "--layers", "2", "--cutoff", "4", "--updates", "2",
                                   "--time-budget-hours", "0"]))
    generated = tmp_path / "gen"
    t.generate(t.parser().parse_args(["generate", "--checkpoint", str(out / "checkpoint.pt"),
        "--output", str(generated), "--device", "cpu", "--steps", "3"]))
    final = ase.io.read(generated / "final.extxyz")
    np.testing.assert_allclose(final.get_masses(), t.BEAD_MASS)
    final_data = ase.io.read(generated / "final.data", format="lammps-data", Z_of_type={1: 14})
    np.testing.assert_allclose(final_data.get_masses(), t.BEAD_MASS)

    report = tmp_path / "eval.json"
    t.evaluate(t.parser().parse_args(["evaluate", "--dataset", str(data), "--sample", str(generated / "final.extxyz"),
        "--output", str(report)]))
    assert report.with_suffix(".png").exists()


def test_train_generate_resume(tmp_path):
    data = dataset(tmp_path)
    common = ["train", "--dataset", str(data), "--device", "cpu", "--width", "8", "--layers", "2",
              "--cutoff", "4", "--log-every", "1", "--checkpoint-every", "1", "--time-budget-hours", "0"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    assert t.train(t.parser().parse_args(common + ["--output", str(full), "--updates", "2"])) == 0
    t.train(t.parser().parse_args(common + ["--output", str(resumed), "--updates", "1"]))
    t.train(t.parser().parse_args(common + ["--output", str(resumed), "--updates", "2", "--resume"]))
    a, b = t.load_pt(full / "checkpoint.pt"), t.load_pt(resumed / "checkpoint.pt")
    for k in a["model"]:
        torch.testing.assert_close(a["model"][k], b["model"][k], rtol=0, atol=0)

    out = tmp_path / "gen"
    args = t.parser().parse_args(["generate", "--checkpoint", str(full / "checkpoint.pt"),
        "--steps", "4", "--device", "cpu", "--output", str(out), "--checkpoint-every", "1"])
    assert t.generate(args) == 0
    trajectory = np.load(out / "positions.npy").copy()
    paused = tmp_path / "paused"
    args.output = paused
    original_save = t.save_pt
    def pause_save(path, obj):
        original_save(path, obj)
        if obj.get("step") == 2:
            t.STOP = True
    t.save_pt = pause_save
    try:
        assert t.generate(args) == 75
    finally:
        t.save_pt, t.STOP = original_save, False
    args.resume = True
    assert t.generate(args) == 0
    np.testing.assert_array_equal(trajectory, np.load(paused / "positions.npy"))


def test_reject_duplicate_only_validation(tmp_path):
    crystal = corner_sharing_crystal()
    path = tmp_path / "same.extxyz"
    ase.io.write(path, [crystal, crystal.copy()])
    args = t.parser().parse_args(["prepare", "--input", str(path), "--output", str(tmp_path / "bad")])
    with pytest.raises(ValueError, match="One unique frame"):
        t.prepare(args)


def test_wrapped_score_matches_image_density_derivative():
    clean = torch.zeros((3, 3), dtype=torch.float64)
    pos = torch.tensor([[0.1, 4.9, 9.9], [5., 2., 3.], [8., 7., 6.]], dtype=torch.float64, requires_grad=True)
    box = torch.tensor([10., 11., 12.], dtype=torch.float64)
    for sigma in (0.1, 1.99, 2., 3., 12.):
        images = pos[..., None] + torch.arange(-12, 13) * box[:, None]
        logdensity = torch.logsumexp(-images.square() / (2 * sigma**2), -1).sum()
        expected = sigma * torch.autograd.grad(logdensity, pos, retain_graph=True)[0]
        actual = t.wrapped_target(pos, clean, box, sigma)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(t.wrapped_target(pos + 2 * box, clean, box, sigma), actual)


def test_periodicity_translation_permutation_and_conditions():
    torch.manual_seed(42)
    model = t.Score(width=8, layers=2, cutoff=4.).double()
    pos = torch.tensor([[1., 1., 1.], [2., 1., 1.], [1.5, 2., 1.4], [1., 1., 2.3]], dtype=torch.float64)
    original = pos.clone()
    types = torch.ones(4, dtype=torch.long)
    box = [12., 13., 14.]
    def predict(x, sigma=.2):
        return model(types, x, t.graph(x, box), sigma, box)
    pred = predict(pos)
    perm = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(predict(pos[perm]), pred[perm])
    shifted = (pos + pos.new_tensor([11., 7., 3.])) % pos.new_tensor(box)
    torch.testing.assert_close(predict(shifted), pred, atol=1e-12, rtol=1e-8)
    torch.testing.assert_close(predict(pos + pos.new_tensor(box) * 2), pred, atol=1e-12, rtol=1e-8)
    assert not torch.allclose(predict(pos, .8), pred)
    pred.square().sum().backward()
    assert model.egnn.embedding_in.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.testing.assert_close(pos, original)


def test_projection_is_periodic_tangent_with_cartesian_units():
    model = t.Score(width=8, layers=1).double()
    box = torch.tensor([12., 13., 14.], dtype=torch.float64)
    pos = torch.tensor([[1., 2., 3.], [3., 4., 5.]], dtype=torch.float64, requires_grad=True)
    z = model.uplift(pos / box)
    fixed = torch.randn_like(z)
    derivative = torch.autograd.grad((z * fixed).sum(), pos)[0]
    projected = torch.einsum("ni,aij,nj->na", z, model.gamma, fixed) / box
    torch.testing.assert_close(derivative, -2 * np.pi * projected)
    torch.testing.assert_close(torch.einsum("ni,aij,nj->na", z, model.gamma, z), torch.zeros_like(pos))


def test_fully_connected_graph_forward_backward():
    model = t.Score(width=8, layers=2)
    pos = torch.tensor([[1., 1., 1.], [7., 7., 7.]])
    out = model(torch.ones(2, dtype=torch.long), pos, t.graph(pos, [14.] * 3), .2, [14.] * 3)
    assert torch.isfinite(out).all()
    out.square().sum().backward()


def test_short_fit_reduces_radial_denoising_loss():
    torch.manual_seed(10)
    model = t.Score(width=16, layers=2, cutoff=4.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    clean = 4 + torch.rand(8, 3) * 2
    types = torch.ones(8, dtype=torch.long)
    sigma, box = .1, [12.] * 3
    # A known relative radial displacement tests learnability without requiring
    # recovery of a fixed absolute origin or arbitrary labeled-atom identities.
    noisy = clean + sigma * .4 * (clean - clean.mean(0))
    edges = t.graph(noisy, box)
    target = t.wrapped_target(noisy, clean, box, sigma)
    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        loss = (model(types, noisy, edges, sigma, box) - target).square().mean()
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * .65


def test_old_checkpoint_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save(dict(format=t.FORMAT), path)
    args = t.parser().parse_args(["generate", "--checkpoint", str(path),
        "--output", str(tmp_path / "gen"), "--device", "cpu"])
    with pytest.raises(ValueError, match="retrain"):
        t.generate(args)


def test_bundled_legacy_dataset_loads():
    from pathlib import Path
    arrays, meta = t.load_dataset(Path(__file__).parent / "examples" / "crystal")
    assert arrays.shape == (62, 64, 3)
    assert meta["format"] == "test42-crystal-single-phase-v1"


def test_extra_noise_schedule():
    levels = t.extra_noise_levels(100, 2.0, 0.1)
    assert levels[0] == 2.0 and np.all(np.diff(levels[:90]) < 0)
    assert np.all(levels[90:] == 0) and levels[89] > 0
    assert np.all(t.extra_noise_levels(10, 0.0, 0.03) == 0)
