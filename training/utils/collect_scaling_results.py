from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and fit scaling-law experiment outputs.")
    parser.add_argument("--root", default="out/scaling", help="Root directory to scan for output.pt files.")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV/JSON outputs. Defaults to --root.")
    parser.add_argument("--output-name", default="output.pt", help="Scaling artifact filename.")
    parser.add_argument(
        "--budget-field",
        choices=("tokens_seen", "target_tokens"),
        default="tokens_seen",
        help="Token budget used for fitting. Grouping still uses target_tokens.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _payload_to_row(path: Path, payload: dict[str, Any], root: Path) -> dict[str, Any]:
    cfg = payload.get("config", {})
    runtime_cfg = cfg.get("runtime", {})
    optimization_cfg = cfg.get("optimization", {})
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "model_name": payload.get("model_name"),
        "seed": runtime_cfg.get("seed"),
        "lr": optimization_cfg.get("lr"),
        "target_tokens": payload.get("target_tokens"),
        "tokens_seen": payload.get("tokens_seen"),
        "token_overshoot": payload.get("token_overshoot"),
        "token_overshoot_ratio": payload.get("token_overshoot_ratio"),
        "params_total": payload.get("params_total"),
        "params_trainable": payload.get("params_trainable"),
        "params_no_embed": payload.get("params_no_embed"),
        "flops_per_token": payload.get("flops_per_token"),
        "declared_compute_flops": payload.get("declared_compute_flops"),
        "actual_execution_flops": payload.get("actual_execution_flops"),
        "best_train_loss": payload.get("best_train_loss"),
        "best_val_loss": payload.get("best_val_loss"),
        "best_val_step": payload.get("best_val_step"),
        "best_val_tokens_seen": payload.get("best_val_tokens_seen"),
        "step": payload.get("step"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _select_best_lr(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        loss = _safe_float(row.get("best_val_loss"))
        target_tokens = row.get("target_tokens")
        seed = row.get("seed")
        model_name = row.get("model_name")
        if loss is None or target_tokens is None or seed is None or model_name is None:
            continue
        key = (str(model_name), int(target_tokens), int(seed))
        current = groups.get(key)
        if current is None or loss < float(current["best_val_loss"]):
            groups[key] = dict(row)
    return sorted(
        groups.values(),
        key=lambda row: (str(row.get("model_name")), int(row.get("target_tokens")), int(row.get("seed"))),
    )


def _fit_loglog(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(mask.sum()) < 2:
        return None
    slope, intercept = np.polyfit(np.log(x[mask]), np.log(y[mask]), deg=1)
    return {"a": float(math.exp(intercept)), "b": float(slope), "num_points": int(mask.sum())}


def _fit_surface(rows: list[dict[str, Any]], *, budget_field: str) -> dict[str, Any]:
    values = []
    for row in rows:
        n = _safe_float(row.get("params_no_embed"))
        d = _safe_float(row.get(budget_field))
        loss = _safe_float(row.get("best_val_loss"))
        if n is not None and d is not None and loss is not None and n > 0 and d > 0 and loss > 0:
            values.append((n, d, loss))
    if len(values) < 5:
        return {"available": False, "reason": "Need at least 5 valid best-LR points.", "num_points": len(values)}
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:
        return {"available": False, "reason": f"scipy is unavailable: {exc}", "num_points": len(values)}

    arr = np.asarray(values, dtype=np.float64)
    n = arr[:, 0]
    d = arr[:, 1]
    loss = arr[:, 2]

    def surface(inputs: tuple[np.ndarray, np.ndarray], e: float, a: float, alpha: float, b: float, beta: float) -> np.ndarray:
        n_values, d_values = inputs
        return e + a * np.power(n_values, -alpha) + b * np.power(d_values, -beta)

    loss_min = float(loss.min())
    p0 = [max(1e-6, loss_min * 0.8), 1.0, 0.1, 1.0, 0.1]
    lower = [0.0, 0.0, 0.0, 0.0, 0.0]
    upper = [loss_min, np.inf, 5.0, np.inf, 5.0]
    try:
        params, _ = curve_fit(
            surface,
            (n, d),
            loss,
            p0=p0,
            bounds=(lower, upper),
            maxfev=100000,
        )
    except Exception as exc:
        return {"available": False, "reason": f"surface fit failed: {exc}", "num_points": len(values)}

    pred = surface((n, d), *params)
    ss_res = float(np.square(loss - pred).sum())
    ss_tot = float(np.square(loss - loss.mean()).sum())
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "available": True,
        "num_points": len(values),
        "formula": "L(N,D) = E + A * N^(-alpha) + B * D^(-beta)",
        "E": float(params[0]),
        "A": float(params[1]),
        "alpha": float(params[2]),
        "B": float(params[3]),
        "beta": float(params[4]),
        "r2": r2,
    }


def _fit_compute_trends(rows: list[dict[str, Any]], *, budget_field: str) -> dict[str, Any]:
    compute_field = "actual_execution_flops" if budget_field == "tokens_seen" else "declared_compute_flops"
    compute = np.asarray([_safe_float(row.get(compute_field)) or np.nan for row in rows], dtype=np.float64)
    lr = np.asarray([_safe_float(row.get("lr")) or np.nan for row in rows], dtype=np.float64)
    params = np.asarray([_safe_float(row.get("params_no_embed")) or np.nan for row in rows], dtype=np.float64)
    tokens = np.asarray([_safe_float(row.get(budget_field)) or np.nan for row in rows], dtype=np.float64)
    loss = np.asarray([_safe_float(row.get("best_val_loss")) or np.nan for row in rows], dtype=np.float64)
    return {
        "budget_field": budget_field,
        "compute_field": compute_field,
        "lr_vs_compute": _fit_loglog(compute, lr),
        "params_vs_compute": _fit_loglog(compute, params),
        "tokens_vs_compute": _fit_loglog(compute, tokens),
        "loss_vs_compute": _fit_loglog(compute, loss),
    }


def _maybe_write_plots(output_dir: Path, rows: list[dict[str, Any]], *, budget_field: str) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    compute_field = "actual_execution_flops" if budget_field == "tokens_seen" else "declared_compute_flops"
    compute = np.asarray([_safe_float(row.get(compute_field)) or np.nan for row in rows], dtype=np.float64)
    loss = np.asarray([_safe_float(row.get("best_val_loss")) or np.nan for row in rows], dtype=np.float64)
    lr = np.asarray([_safe_float(row.get("lr")) or np.nan for row in rows], dtype=np.float64)
    model_names = [str(row.get("model_name")) for row in rows]
    written = []

    mask = np.isfinite(compute) & np.isfinite(loss) & (compute > 0) & (loss > 0)
    if int(mask.sum()) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model_name in sorted(set(model_names)):
            model_mask = mask & np.asarray([name == model_name for name in model_names], dtype=bool)
            if int(model_mask.sum()) == 0:
                continue
            ax.scatter(compute[model_mask], loss[model_mask], label=model_name)
        ax.set_xscale("log")
        ax.set_xlabel(f"{compute_field} FLOPs")
        ax.set_ylabel("Best validation loss")
        ax.set_title(f"Loss vs {compute_field}")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "loss_vs_compute.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    mask = np.isfinite(compute) & np.isfinite(lr) & (compute > 0) & (lr > 0)
    if int(mask.sum()) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(compute[mask], lr[mask])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{compute_field} FLOPs")
        ax.set_ylabel("Selected learning rate")
        ax.set_title(f"Best LR vs {compute_field}")
        fig.tight_layout()
        path = output_dir / "lr_vs_compute.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    return written


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root
    paths = sorted(root.rglob(str(args.output_name)))
    rows = []
    for path in paths:
        try:
            payload = _load_payload(path)
            rows.append(_payload_to_row(path, payload, root))
        except Exception as exc:
            print(f"[COLLECT] skipping {path}: {exc}")

    best_rows = _select_best_lr(rows)
    _write_csv(output_dir / "scaling_results.csv", rows)
    _write_jsonl(output_dir / "scaling_results.jsonl", rows)
    _write_csv(output_dir / "scaling_best_lr_results.csv", best_rows)

    fit = {
        "num_runs": len(rows),
        "num_best_lr_runs": len(best_rows),
        "budget_field": args.budget_field,
        "surface": _fit_surface(best_rows, budget_field=args.budget_field),
        "compute_trends": _fit_compute_trends(best_rows, budget_field=args.budget_field),
    }
    plots = _maybe_write_plots(output_dir, best_rows, budget_field=args.budget_field)
    fit["plots"] = plots
    with (output_dir / "scaling_fit.json").open("w", encoding="utf-8") as f:
        json.dump(fit, f, indent=2, sort_keys=True)

    print(f"[COLLECT] runs={len(rows)} best_lr_runs={len(best_rows)} output_dir={output_dir} plots={len(plots)}")


if __name__ == "__main__":
    main()
