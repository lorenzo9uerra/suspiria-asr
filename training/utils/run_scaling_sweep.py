from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from training.utils.config import to_plain_dict


def _validate_model_grid(cfg: dict[str, Any]) -> None:
    sweep_cfg = cfg["sweep"]
    head_dim = int(sweep_cfg["head_dim"])
    heads_per_kv_head = int(sweep_cfg["heads_per_kv_head"])
    models = sweep_cfg["models"]
    for name in sweep_cfg["enabled_models"]:
        if name not in models:
            raise KeyError(f"Enabled model {name!r} is not defined in sweep.models.")
        model = models[name]
        hidden_size = int(model["hidden_size"])
        num_heads = int(model["num_heads"])
        num_kv_heads = int(model["num_kv_heads"])
        if hidden_size != num_heads * head_dim:
            raise ValueError(
                f"Invalid model {name!r}: hidden_size={hidden_size} must equal "
                f"num_heads * head_dim = {num_heads} * {head_dim}."
            )
        if num_heads != num_kv_heads * heads_per_kv_head:
            raise ValueError(
                f"Invalid model {name!r}: num_heads={num_heads} must equal "
                f"num_kv_heads * heads_per_kv_head = {num_kv_heads} * {heads_per_kv_head}."
            )


def _format_lr(value: float) -> str:
    return f"{float(value):.8g}".replace(".", "p").replace("-", "m")


def _build_command(
    *,
    cfg: dict[str, Any],
    model_name: str,
    target_tokens: int,
    lr: float,
    seed: int,
) -> list[str]:
    sweep_cfg = cfg["sweep"]
    model = sweep_cfg["models"][model_name]
    output_dir = (
        Path(str(sweep_cfg["output_root"]))
        / str(model_name)
        / f"D_{int(target_tokens)}"
        / f"lr_{_format_lr(float(lr))}"
        / f"seed_{int(seed)}"
    )
    return [
        sys.executable,
        "-m",
        "training.train",
        "scaling.enabled=true",
        f"scaling.model_name={model_name}",
        f"scaling.target_tokens={int(target_tokens)}",
        f"optimization.lr={float(lr)}",
        f"runtime.seed={int(seed)}",
        f"runtime.output_dir={str(output_dir)}",
        f"model.hidden_size={int(model['hidden_size'])}",
        f"model.num_layers={int(model['num_layers'])}",
        f"model.num_heads={int(model['num_heads'])}",
        f"model.num_kv_heads={int(model['num_kv_heads'])}",
        f"model.ffw_hidden_size={int(model['ffw_hidden_size'])}",
    ]


def iter_commands(cfg: dict[str, Any]) -> list[list[str]]:
    commands = []
    for model_name in cfg["sweep"]["enabled_models"]:
        for target_tokens in cfg["sweep"]["target_tokens"]:
            for lr in cfg["sweep"]["learning_rates"]:
                for seed in cfg["sweep"]["seeds"]:
                    commands.append(
                        _build_command(
                            cfg=cfg,
                            model_name=str(model_name),
                            target_tokens=int(target_tokens),
                            lr=float(lr),
                            seed=int(seed),
                        )
                    )
    return commands


@hydra.main(version_base=None, config_path="../../configs", config_name="scaling")
def main(cfg: DictConfig) -> None:
    cfg = to_plain_dict(cfg)
    _validate_model_grid(cfg)
    commands = iter_commands(cfg)
    print(f"[SWEEP] commands={len(commands)} run_serial={bool(cfg['sweep'].get('run_serial', False))}")
    for command in commands:
        print(shlex.join(command))
    if not bool(cfg["sweep"].get("run_serial", False)):
        return
    for idx, command in enumerate(commands, start=1):
        print(f"[SWEEP] running {idx}/{len(commands)}: {shlex.join(command)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
