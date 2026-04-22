from __future__ import annotations

import hydra
import math
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from training.data.collator import SpecialTokenIds
from training.tokenizer import load_tokenizer
from training.utils.checkpointing import (
    maybe_build_ema,
    maybe_resume_training_state,
    save_tokenizer_artifacts,
    save_training_state,
)
from training.utils.config import resolve_device, set_random_seeds, to_plain_dict
from training.utils.data import (
    build_dataloader,
    build_raw_dataloader,
    discover_materialized_splits,
    ensure_materialized_dataset,
)
from training.data.materialize_latents import resolve_manifest_root
from training.utils.evaluation import evaluate_loss, evaluate_wer, select_eval_model
from training.utils.metrics import (
    compute_batch_metric_counts,
    finalize_metric_counts,
    merge_metric_counts,
)
from training.utils.model_builder import build_model, load_pretrained_model_weights
from training.utils.optimization import build_optimizer_and_scheduler
from training.utils.scaling import build_scaling_payload, save_scaling_output
from training.utils.logging import print_model_parameter_summary, silence_external_info_logs

try:
    import wandb
except Exception:
    wandb = None


OmegaConf.register_new_resolver(
    "path_token",
    lambda value: str(value).replace(".", "p").replace("-", "m"),
    replace=True,
)

silence_external_info_logs()


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _wer_summary(metrics: dict[str, float]) -> str:
    if "wer" in metrics:
        return f" wer={metrics['wer']:.4f}"
    values = [
        f"{key}={value:.4f}"
        for key, value in sorted(metrics.items())
        if key.startswith("wer/delay_")
    ]
    if not values:
        return ""
    return " " + " ".join(values)


def _is_missing(value: object) -> bool:
    return value in (None, "", "null", "None")


def _resolve_scaling_target(scaling_cfg: dict[str, object], *, enabled: bool) -> int | None:
    target_tokens = scaling_cfg.get("target_tokens")
    if not enabled:
        return None
    if _is_missing(target_tokens):
        raise ValueError("scaling.enabled=true requires scaling.target_tokens to be set.")
    target_tokens_int = int(target_tokens)
    if target_tokens_int <= 0:
        raise ValueError(f"scaling.target_tokens must be positive, got {target_tokens!r}.")
    return target_tokens_int


def _resolve_requested_splits(dataset_cfg: dict[str, object]) -> set[str]:
    raw_splits = dataset_cfg.get("splits", ("train", "validation", "test"))
    if _is_missing(raw_splits):
        return {"train", "validation", "test"}
    if isinstance(raw_splits, str):
        requested = {raw_splits}
    else:
        requested = {str(split) for split in raw_splits}
    valid = {"train", "validation", "test"}
    unknown = requested - valid
    if unknown:
        raise ValueError(f"Unsupported dataset.splits entries: {sorted(unknown)}. Expected only {sorted(valid)}.")
    return requested


def _estimate_scaling_total_steps(cfg: dict[str, object], target_tokens: int) -> int:
    dataset_cfg = cfg["dataset"]
    scaling_cfg = cfg.get("scaling", {})
    avg_audio_seconds = float(scaling_cfg.get("avg_audio_seconds", 15.0))
    step_ms = int(dataset_cfg.get("step_ms", 80))
    left_pad_steps = int(dataset_cfg.get("left_pad_steps", 0))
    delay_min_ms = int(dataset_cfg.get("delay_min_ms", 80))
    delay_max_ms = int(dataset_cfg.get("delay_max_ms", delay_min_ms))
    avg_delay_steps = ((delay_min_ms + delay_max_ms) / 2.0) / float(step_ms)
    avg_real_steps = avg_audio_seconds * 1000.0 / float(step_ms)
    avg_tokens_per_sample = max(1.0, avg_real_steps + left_pad_steps + avg_delay_steps)
    tokens_per_step = max(1.0, avg_tokens_per_sample * int(cfg["optimization"].get("batch_size", 1)))
    return max(1, int(math.ceil(float(target_tokens) / tokens_per_step)))


@hydra.main(version_base=None, config_path="../configs", config_name="training")
def main(cfg: DictConfig) -> None:
    cfg = to_plain_dict(cfg)
    scaling_cfg = cfg.get("scaling", {})
    scaling_enabled = bool(scaling_cfg.get("enabled", False))
    target_tokens_int = _resolve_scaling_target(scaling_cfg, enabled=scaling_enabled)

    set_random_seeds(
        int(cfg["runtime"].get("seed", 1337)),
        deterministic=bool(cfg["runtime"].get("deterministic", False)),
    )

    resolved_tokenizer = load_tokenizer(cfg["tokenizer"])
    tokenizer = resolved_tokenizer.tokenizer
    special_tokens = SpecialTokenIds(
        bos=resolved_tokenizer.bos_token_id,
        eos=resolved_tokenizer.eos_token_id,
        pad_wait=resolved_tokenizer.pad_wait_token_id,
        word_start=resolved_tokenizer.word_start_token_id,
    )

    materialized_root = ensure_materialized_dataset(cfg)
    device = resolve_device(cfg["runtime"])
    test_only = bool(cfg["runtime"].get("test_only", False))
    country = str(cfg["dataset"]["country"])
    manifest_root = resolve_manifest_root(cfg["dataset"])
    available_split_names = discover_materialized_splits(
        manifest_root=manifest_root,
        country=country,
    )
    requested_splits = _resolve_requested_splits(cfg["dataset"])
    split_names = available_split_names & requested_splits
    print(f"[DATA] available_splits={sorted(available_split_names)}")
    print(f"[DATA] requested_splits={sorted(requested_splits)} active_splits={sorted(split_names)}")
    wer_enabled = bool(cfg.get("wer", {}).get("enabled", False))

    train_loader = None
    if not test_only:
        if "train" not in split_names:
            raise FileNotFoundError(
                f"Manifest-defined train split not found for country={country!r}."
            )
        train_loader = build_dataloader(
            cfg=cfg,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            materialized_root=materialized_root,
            split="train",
            manifest_root=manifest_root,
        )
    val_loader = None
    val_wer_loader = None
    if not test_only and "validation" in split_names:
        val_loader = build_dataloader(
            cfg=cfg,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            materialized_root=materialized_root,
            split="validation",
            manifest_root=manifest_root,
        )
        if wer_enabled:
            val_wer_loader = build_raw_dataloader(
                cfg=cfg,
                materialized_root=materialized_root,
                split="validation",
                manifest_root=manifest_root,
            )
    if scaling_enabled and val_loader is None:
        raise ValueError("Scaling-law runs require an active validation split.")
    test_loader = None
    test_wer_loader = None
    if "test" in split_names:
        test_loader = build_dataloader(
            cfg=cfg,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            materialized_root=materialized_root,
            split="test",
            manifest_root=manifest_root,
        )
        if wer_enabled:
            test_wer_loader = build_raw_dataloader(
                cfg=cfg,
                materialized_root=materialized_root,
                split="test",
                manifest_root=manifest_root,
            )

    model = build_model(
        cfg,
        vocab_size=len(tokenizer),
        device=device,
        special_tokens=special_tokens,
    )

    pretrained_weights_path = cfg["model"].get("pretrained_weights_path")
    if pretrained_weights_path not in (None, "", "null", "None"):
        missing, unexpected = load_pretrained_model_weights(
            model,
            weights_path=str(pretrained_weights_path),
            strict=bool(cfg["model"].get("pretrained_strict", False)),
        )
        if missing or unexpected:
            print(f"[INIT] pretrained missing={missing} unexpected={unexpected}")

    print_model_parameter_summary(model)

    optimizer_train_cfg = dict(cfg["optimization"])
    estimated_scheduler_max_steps = None
    if scaling_enabled:
        assert target_tokens_int is not None
        estimated_scheduler_max_steps = _estimate_scaling_total_steps(cfg, target_tokens_int)
        optimizer_train_cfg["max_steps"] = estimated_scheduler_max_steps
        print(f"[SCALING] estimated_scheduler_max_steps={estimated_scheduler_max_steps}")

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        device=device,
        train_cfg=optimizer_train_cfg,
    )
    ema = maybe_build_ema(model, cfg)
    if scaling_enabled and not _is_missing(cfg["runtime"].get("checkpoint_path")):
        raise ValueError("Scaling-law runs do not support runtime.checkpoint_path resume.")
    start_step, _, _ = maybe_resume_training_state(
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
    )
    if start_step > 0:
        print(f"[RESUME] Resuming training from step {start_step}")

    save_tokenizer_artifacts(tokenizer, special_tokens, cfg)

    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled:
        if wandb is None:
            raise ImportError("wandb logging is enabled but the `wandb` package is not installed.")
        wandb.init(
            project=str(wandb_cfg["project"]),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            group=wandb_cfg.get("group"),
            tags=list(wandb_cfg.get("tags", [])),
            mode=str(wandb_cfg.get("mode", "online")),
            config=cfg,
            dir=cfg["runtime"].get("output_dir"),
        )
        wandb.summary["country"] = country
        wandb.summary["device"] = str(device)
        if scaling_enabled:
            wandb.summary["scaling/model_name"] = scaling_cfg.get("model_name")
            wandb.summary["scaling/target_tokens"] = target_tokens_int

    eval_cfg = cfg.get("evaluation", {})
    eval_max_batches = eval_cfg.get("max_eval_batches")
    compute_train_metrics = bool(eval_cfg.get("compute_train_metrics", False))
    if eval_max_batches in (None, "null"):
        eval_max_batches = None
    else:
        eval_max_batches = int(eval_max_batches)

    if test_only:
        if test_loader is None:
            raise ValueError("runtime.test_only=true but no test split was found in the materialized dataset.")
        eval_model = select_eval_model(model, ema=ema, cfg=cfg)
        test_metrics = evaluate_loss(
            eval_model,
            test_loader,
            device=device,
            special_tokens=special_tokens,
            max_batches=eval_max_batches,
        )
        if wer_enabled and test_wer_loader is not None:
            test_metrics.update(
                evaluate_wer(
                    eval_model,
                    test_wer_loader,
                    tokenizer=tokenizer,
                    special_tokens=special_tokens,
                    device=device,
                    cfg=cfg,
                )
            )
        print(
            "[TEST] "
            f"loss={test_metrics['loss']:.4f} "
            f"unweighted_loss={test_metrics['unweighted_loss']:.4f} "
            f"perplexity={test_metrics['perplexity']:.4f} "
            f"batches={int(test_metrics['num_batches'])}"
            f"{_wer_summary(test_metrics)}"
        )
        if wandb_enabled:
            wandb.log(_prefix_metrics("test", test_metrics), step=start_step)
            wandb.finish()
        return

    max_steps = int(cfg["optimization"].get("max_steps", 1000))
    save_every = int(cfg["runtime"].get("save_every_steps", 100))
    log_every = int(cfg["runtime"].get("log_every_steps", 10))
    validation_every = int(eval_cfg.get("validation_every_steps", save_every))
    grad_clip = float(cfg["optimization"].get("grad_clip_norm", 1.0))

    assert train_loader is not None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    log_window_start = time.perf_counter()
    log_window_tokens = 0
    log_window_samples = 0
    log_window_metric_counts = None
    tokens_seen = 0
    observed_max_seq_len = 0
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_val_step = None
    best_val_tokens_seen = None
    best_val_metrics = None
    final_val_metrics = None
    final_test_metrics = None

    while step < max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch)
            outputs["loss"].backward()
            grad_norm = None
            if grad_clip > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).detach().cpu()
                )
            elif compute_train_metrics:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).detach().cpu()
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)

            step += 1
            batch_tokens = int(batch["packed_labels"].numel())
            tokens_seen += batch_tokens
            observed_max_seq_len = max(observed_max_seq_len, int(batch["max_seq_len"].detach().cpu().item()))
            current_train_unweighted_loss = float(outputs["unweighted_loss"].detach().cpu())
            best_train_loss = min(best_train_loss, current_train_unweighted_loss)
            log_window_tokens += batch_tokens
            log_window_samples += int(batch["seq_lens"].shape[0])
            if compute_train_metrics:
                batch_counts = compute_batch_metric_counts(
                    outputs["logits"],
                    batch["packed_labels"],
                    special_tokens=special_tokens,
                    loss_value=float(outputs["loss"].detach().cpu()),
                    unweighted_loss_value=float(outputs["unweighted_loss"].detach().cpu()),
                    loss_sum=float(outputs["loss_sum"].detach().cpu()),
                    loss_weight_sum=float(outputs["loss_weight_sum"].detach().cpu()),
                    unweighted_loss_sum=float(outputs["unweighted_loss_sum"].detach().cpu()),
                    token_count=int(outputs["token_count"].detach().cpu()),
                )
                if log_window_metric_counts is None:
                    log_window_metric_counts = batch_counts
                else:
                    merge_metric_counts(log_window_metric_counts, batch_counts)
            if step % log_every == 0:
                lr = float(optimizer.param_groups[0]["lr"])
                elapsed = max(time.perf_counter() - log_window_start, 1e-6)
                train_log = {
                    "train/loss": float(outputs["loss"].detach().cpu()),
                    "train/unweighted_loss": current_train_unweighted_loss,
                    "train/perplexity": float(torch.exp(outputs["unweighted_loss"].detach().cpu()).item()),
                    "train/lr": lr,
                    "train/grad_norm": float("nan") if grad_norm is None else grad_norm,
                    "train/tokens_per_sec": float(log_window_tokens) / elapsed,
                    "train/samples_per_sec": float(log_window_samples) / elapsed,
                    "train/step": step,
                }
                if scaling_enabled:
                    train_log["scaling/tokens_seen"] = float(tokens_seen)
                    train_log["scaling/target_tokens"] = float(target_tokens_int)
                    train_log["scaling/token_progress"] = float(tokens_seen) / float(target_tokens_int)
                if compute_train_metrics and log_window_metric_counts is not None:
                    train_log.update(
                        _prefix_metrics(
                            "train",
                            finalize_metric_counts(log_window_metric_counts),
                        )
                    )
                print(
                    f"step={step} loss={train_log['train/loss']:.4f} "
                    f"lr={lr:.8f} "
                    f"tokens_per_sec={train_log['train/tokens_per_sec']:.2f} "
                    f"samples_per_sec={train_log['train/samples_per_sec']:.2f}"
                )
                if wandb_enabled:
                    wandb.log(train_log, step=step)
                log_window_start = time.perf_counter()
                log_window_tokens = 0
                log_window_samples = 0
                log_window_metric_counts = None
            if val_loader is not None and step % validation_every == 0:
                eval_model = select_eval_model(model, ema=ema, cfg=cfg)
                val_metrics = evaluate_loss(
                    eval_model,
                    val_loader,
                    device=device,
                    special_tokens=special_tokens,
                    max_batches=eval_max_batches,
                )
                if wer_enabled and val_wer_loader is not None:
                    val_metrics.update(
                        evaluate_wer(
                            eval_model,
                            val_wer_loader,
                            tokenizer=tokenizer,
                            special_tokens=special_tokens,
                            device=device,
                            cfg=cfg,
                        )
                    )
                print(
                    "[VAL] "
                    f"step={step} "
                    f"loss={val_metrics['loss']:.4f} "
                    f"unweighted_loss={val_metrics['unweighted_loss']:.4f} "
                    f"perplexity={val_metrics['perplexity']:.4f} "
                    f"batches={int(val_metrics['num_batches'])}"
                    f"{_wer_summary(val_metrics)}"
                )
                if wandb_enabled:
                    wandb.log(_prefix_metrics("val", val_metrics), step=step)
                val_unweighted_loss = float(val_metrics["unweighted_loss"])
                if val_unweighted_loss < best_val_loss:
                    best_val_loss = val_unweighted_loss
                    best_val_step = step
                    best_val_tokens_seen = tokens_seen
                    best_val_metrics = dict(val_metrics)
            if step % save_every == 0:
                save_training_state(
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    step=step,
                )
            reached_token_budget = scaling_enabled and tokens_seen >= int(target_tokens_int)
            if step >= max_steps or reached_token_budget:
                break

    save_training_state(
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        step=step,
    )
    if val_loader is not None:
        eval_model = select_eval_model(model, ema=ema, cfg=cfg)
        final_val_metrics = evaluate_loss(
            eval_model,
            val_loader,
            device=device,
            special_tokens=special_tokens,
            max_batches=eval_max_batches,
        )
        if wer_enabled and val_wer_loader is not None:
            final_val_metrics.update(
                evaluate_wer(
                    eval_model,
                    val_wer_loader,
                    tokenizer=tokenizer,
                    special_tokens=special_tokens,
                    device=device,
                    cfg=cfg,
                )
            )
        print(
            "[VAL] "
            f"final_step={step} "
            f"loss={final_val_metrics['loss']:.4f} "
            f"unweighted_loss={final_val_metrics['unweighted_loss']:.4f} "
            f"perplexity={final_val_metrics['perplexity']:.4f} "
            f"batches={int(final_val_metrics['num_batches'])}"
            f"{_wer_summary(final_val_metrics)}"
        )
        val_unweighted_loss = float(final_val_metrics["unweighted_loss"])
        if val_unweighted_loss < best_val_loss:
            best_val_loss = val_unweighted_loss
            best_val_step = step
            best_val_tokens_seen = tokens_seen
            best_val_metrics = dict(final_val_metrics)
        if wandb_enabled:
            wandb.log(_prefix_metrics("val/final", final_val_metrics), step=step)
    if test_loader is not None:
        eval_model = select_eval_model(model, ema=ema, cfg=cfg)
        final_test_metrics = evaluate_loss(
            eval_model,
            test_loader,
            device=device,
            special_tokens=special_tokens,
            max_batches=eval_max_batches,
        )
        if wer_enabled and test_wer_loader is not None:
            final_test_metrics.update(
                evaluate_wer(
                    eval_model,
                    test_wer_loader,
                    tokenizer=tokenizer,
                    special_tokens=special_tokens,
                    device=device,
                    cfg=cfg,
                )
            )
        print(
            "[TEST] "
            f"final_step={step} "
            f"loss={final_test_metrics['loss']:.4f} "
            f"unweighted_loss={final_test_metrics['unweighted_loss']:.4f} "
            f"perplexity={final_test_metrics['perplexity']:.4f} "
            f"batches={int(final_test_metrics['num_batches'])}"
            f"{_wer_summary(final_test_metrics)}"
        )
        if wandb_enabled:
            wandb.log(_prefix_metrics("test", final_test_metrics), step=step)
    if scaling_enabled:
        scaling_output_name = str(scaling_cfg.get("output_name", "output.pt"))
        output_path = Path(cfg["runtime"].get("output_dir", "out/training")).expanduser().resolve() / scaling_output_name
        payload = build_scaling_payload(
            model=model,
            cfg=cfg,
            step=step,
            target_tokens=target_tokens_int,
            tokens_seen=tokens_seen,
            observed_max_seq_len=observed_max_seq_len,
            best_train_loss=None if best_train_loss == float("inf") else best_train_loss,
            best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
            best_val_step=best_val_step,
            best_val_tokens_seen=best_val_tokens_seen,
            estimated_scheduler_max_steps=estimated_scheduler_max_steps,
            best_val_metrics=best_val_metrics,
            final_val_metrics=final_val_metrics,
            final_test_metrics=final_test_metrics,
        )
        save_scaling_output(output_path, payload)
        if wandb_enabled:
            wandb.summary["scaling/tokens_seen"] = tokens_seen
            wandb.summary["scaling/token_overshoot"] = payload["token_overshoot"]
            wandb.summary["scaling/declared_compute_flops"] = payload["declared_compute_flops"]
            wandb.summary["scaling/flops_per_token"] = payload["flops_per_token"]
        print(
            "[SCALING] "
            f"wrote={output_path} "
            f"target_tokens={target_tokens_int} "
            f"tokens_seen={tokens_seen} "
            f"overshoot={payload['token_overshoot']}"
        )
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
