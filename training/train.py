from __future__ import annotations

import hydra
import time
import torch
from omegaconf import DictConfig

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
    discover_materialized_splits,
    ensure_materialized_dataset,
)
from training.data.materialize_latents import resolve_manifest_root
from training.utils.evaluation import evaluate_loss, select_eval_model
from training.utils.metrics import (
    compute_batch_metric_counts,
    finalize_metric_counts,
    merge_metric_counts,
)
from training.utils.model_builder import build_model, load_pretrained_model_weights
from training.utils.optimization import build_optimizer_and_scheduler

try:
    import wandb
except Exception:
    wandb = None


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


@hydra.main(version_base=None, config_path="../configs", config_name="training")
def main(cfg: DictConfig) -> None:
    cfg = to_plain_dict(cfg)
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
    split_names = discover_materialized_splits(
        manifest_root=manifest_root,
        country=country,
    )

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
    if not test_only and "validation" in split_names:
        val_loader = build_dataloader(
            cfg=cfg,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            materialized_root=materialized_root,
            split="validation",
            manifest_root=manifest_root,
        )
    test_loader = None
    if "test" in split_names:
        test_loader = build_dataloader(
            cfg=cfg,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
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

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        device=device,
        train_cfg=cfg["optimization"],
    )
    ema = maybe_build_ema(model, cfg)
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
        print(
            "[TEST] "
            f"loss={test_metrics['loss']:.4f} "
            f"unweighted_loss={test_metrics['unweighted_loss']:.4f} "
            f"perplexity={test_metrics['perplexity']:.4f} "
            f"batches={int(test_metrics['num_batches'])}"
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
            log_window_tokens += int((batch["packed_labels"] != -100).sum().item())
            log_window_samples += int(batch["seq_lens"].shape[0])
            if compute_train_metrics:
                batch_counts = compute_batch_metric_counts(
                    outputs["logits"],
                    batch["packed_labels"],
                    special_tokens=special_tokens,
                    loss_value=float(outputs["loss"].detach().cpu()),
                    unweighted_loss_value=float(outputs["unweighted_loss"].detach().cpu()),
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
                    "train/unweighted_loss": float(outputs["unweighted_loss"].detach().cpu()),
                    "train/perplexity": float(torch.exp(outputs["loss"].detach().cpu()).item()),
                    "train/lr": lr,
                    "train/grad_norm": float("nan") if grad_norm is None else grad_norm,
                    "train/tokens_per_sec": float(log_window_tokens) / elapsed,
                    "train/samples_per_sec": float(log_window_samples) / elapsed,
                    "train/step": step,
                }
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
                print(
                    "[VAL] "
                    f"step={step} "
                    f"loss={val_metrics['loss']:.4f} "
                    f"unweighted_loss={val_metrics['unweighted_loss']:.4f} "
                    f"perplexity={val_metrics['perplexity']:.4f} "
                    f"batches={int(val_metrics['num_batches'])}"
                )
                if wandb_enabled:
                    wandb.log(_prefix_metrics("val", val_metrics), step=step)
            if step % save_every == 0:
                save_training_state(
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    step=step,
                )
            if step >= max_steps:
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
        val_metrics = evaluate_loss(
            eval_model,
            val_loader,
            device=device,
            special_tokens=special_tokens,
            max_batches=eval_max_batches,
        )
        print(
            "[VAL] "
            f"final_step={step} "
            f"loss={val_metrics['loss']:.4f} "
            f"unweighted_loss={val_metrics['unweighted_loss']:.4f} "
            f"perplexity={val_metrics['perplexity']:.4f} "
            f"batches={int(val_metrics['num_batches'])}"
        )
        if wandb_enabled:
            wandb.log(_prefix_metrics("val/final", val_metrics), step=step)
    if test_loader is not None:
        eval_model = select_eval_model(model, ema=ema, cfg=cfg)
        test_metrics = evaluate_loss(
            eval_model,
            test_loader,
            device=device,
            special_tokens=special_tokens,
            max_batches=eval_max_batches,
        )
        print(
            "[TEST] "
            f"final_step={step} "
            f"loss={test_metrics['loss']:.4f} "
            f"unweighted_loss={test_metrics['unweighted_loss']:.4f} "
            f"perplexity={test_metrics['perplexity']:.4f} "
            f"batches={int(test_metrics['num_batches'])}"
        )
        if wandb_enabled:
            wandb.log(_prefix_metrics("test", test_metrics), step=step)
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
