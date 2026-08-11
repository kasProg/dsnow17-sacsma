"""Config-driven training entrypoint (Hydra) for both models this repo
trains: the hybrid Snow17+SAC-SMA+ParamNet stack (CLAUDE.md's Day 7-10
milestone, src/paramnet.py) and the pure-LSTM benchmark
(src/benchmark_lstm.py) used to quantify what the physical constraint
buys (see results/README.md). One script, one CLI -- `model=` switches
which one trains, `split=` switches spatial (prediction in ungauged
basins) vs. temporal (prediction in ungauged period) evaluation. See
configs/ and the main README's "Reproducing experiments" section.

    .venv/bin/python src/train.py                                    # hybrid, spatial (default)
    .venv/bin/python src/train.py model=benchmark_lstm train=benchmark_lstm
    .venv/bin/python src/train.py split=temporal seed=1

Kept as a plain function (run_training(cfg)) wrapped by a thin
@hydra.main CLI (cli()) at the bottom -- tests and results/README.md's
regeneration snippets call run_training() directly with a manually built
config, no Hydra compose/multirun machinery needed for that path.

One gradient step per epoch, not per basin (hybrid model only -- the
benchmark model already batches every basin through one LSTM forward
pass): theta_A/theta_B are computed for all training basins in a single
batched ParamNet forward pass, then each basin's own (expensive,
Fortran-backed) CoupledTwoStageFunction call runs individually -- that
part isn't batchable, SAC-SMA/Snow17 are single-HRU by construction --
and their losses are averaged into ONE scalar before a single
.backward()/optimizer.step() call. Standard full-batch gradient descent
over basins, not per-basin SGD.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from benchmark_lstm import build_normalized_dynamic_arrays  # noqa: E402
from data_module import BasinExample, build_split, masked_nse_loss, nse_value  # noqa: E402
from model_factory import build_model  # noqa: E402
from pipeline import CoupledNWSStack  # noqa: E402


def run_epoch_hybrid(
    net,
    stack: CoupledNWSStack,
    basins: list[BasinExample],
    X_static: dict,
    X_climate: dict,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """optimizer=None -> eval mode, no gradient step. Returns
    {gauge_id: nse}."""
    x_static_batch = torch.tensor(
        np.stack([X_static[b.gauge_id] for b in basins]), dtype=torch.float64
    )
    x_climate_batch = torch.tensor(
        np.stack([X_climate[b.gauge_id] for b in basins]), dtype=torch.float64
    )
    if optimizer is not None:
        net.train()
        theta_A_batch, theta_B_batch = net(x_static_batch, x_climate_batch)
    else:
        net.eval()
        with torch.no_grad():
            theta_A_batch, theta_B_batch = net(x_static_batch, x_climate_batch)
        theta_A_batch = theta_A_batch.detach().requires_grad_(False)
        theta_B_batch = theta_B_batch.detach().requires_grad_(False)

    losses = []
    nses = {}
    for i, ex in enumerate(basins):
        sim = stack.run(theta_A_batch[i], theta_B_batch[i], ex.snow17_forcing, ex.sacsma_forcing)
        loss = masked_nse_loss(sim, ex)
        losses.append(loss)
        nses[ex.gauge_id] = nse_value(sim, ex)

    if optimizer is not None:
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

    return nses


def run_epoch_benchmark(
    net,
    basins: list[BasinExample],
    X_static: dict,
    dynamic_arrays: dict,
    dyn_mean: np.ndarray,
    dyn_std: np.ndarray,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    x_dynamic = torch.tensor(
        np.stack([(dynamic_arrays[b.gauge_id] - dyn_mean) / dyn_std for b in basins]),
        dtype=torch.float64,
    )
    x_static = torch.tensor(np.stack([X_static[b.gauge_id] for b in basins]), dtype=torch.float64)

    if optimizer is not None:
        net.train()
        q_hat = net(x_dynamic, x_static)
    else:
        net.eval()
        with torch.no_grad():
            q_hat = net(x_dynamic, x_static)

    losses = []
    nses = {}
    for i, ex in enumerate(basins):
        sim = q_hat[i]
        loss = masked_nse_loss(sim, ex)
        losses.append(loss)
        with torch.no_grad():
            nses[ex.gauge_id] = float(1.0 - loss.item())

    if optimizer is not None:
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

    return nses


def run_training(cfg: DictConfig) -> dict:
    torch.manual_seed(cfg.seed)  # reproducible network init -- see notes/logs.md

    print(f"Loading basins (data={cfg.data.name}, split={cfg.split.mode})...")
    t0 = time.time()
    split = build_split(cfg)
    print(
        f"  {len(split.train_ids)} train + {len(split.test_ids)} test basins, "
        f"loaded in {time.time()-t0:.1f}s"
    )

    n_static = split.X_static[split.train_ids[0]].shape[0]
    optimizer_lr = cfg.train.lr

    if cfg.model.name == "hybrid":
        n_climate = split.X_climate[split.train_ids[0]].shape[1]
        net = build_model(cfg, n_static, n_climate)
        stack = CoupledNWSStack()
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_lr)

        def run_epoch(basins, opt):
            return run_epoch_hybrid(net, stack, basins, split.X_static, split.X_climate, opt)

    elif cfg.model.name == "benchmark_lstm":
        dynamic_arrays, dyn_mean, dyn_std = build_normalized_dynamic_arrays(
            split.train_examples + split.test_examples, split.train_ids
        )
        net = build_model(cfg, n_static, dynamic_arrays[split.train_ids[0]].shape[1])
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_lr)

        def run_epoch(basins, opt):
            return run_epoch_benchmark(
                net, basins, split.X_static, dynamic_arrays, dyn_mean, dyn_std, opt
            )

    else:
        raise ValueError(f"unknown model.name: {cfg.model.name!r}")

    history = []
    for epoch in range(1, cfg.train.n_epochs + 1):
        t0 = time.time()
        train_nses = run_epoch(split.train_examples, optimizer)
        mean_train_nse = float(np.mean(list(train_nses.values())))
        dt = time.time() - t0

        row = {"epoch": epoch, "mean_train_nse": mean_train_nse, "seconds": dt}
        if epoch == 1 or epoch % cfg.train.eval_every == 0 or epoch == cfg.train.n_epochs:
            test_nses = run_epoch(split.test_examples, None)
            row["mean_test_nse"] = float(np.mean(list(test_nses.values())))
        history.append(row)
        print(
            f"epoch {epoch:3d}  train_nse={mean_train_nse:+.4f}"
            + (f"  test_nse={row.get('mean_test_nse'):+.4f}" if "mean_test_nse" in row else "")
            + f"  ({dt:.2f}s)"
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    torch.save(net.state_dict(), output_dir / "checkpoint.pt")

    result = {
        "model": cfg.model.name,
        "split_mode": cfg.split.mode,
        "seed": cfg.seed,
        "n_epochs": cfg.train.n_epochs,
        "lr": cfg.train.lr,
        "n_train_basins": len(split.train_ids),
        "n_test_basins": len(split.test_ids),
        "train_basin_ids": split.train_ids,
        "test_basin_ids": split.test_ids,
        "history": history,
    }
    (output_dir / "history.json").write_text(json.dumps(result, indent=2))
    print(f"Saved config + checkpoint + history -> {output_dir}")

    return {"net": net, "output_dir": str(output_dir), **result}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def cli(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    cli()
