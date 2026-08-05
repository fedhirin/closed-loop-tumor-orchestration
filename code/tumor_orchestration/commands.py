from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from tumor_orchestration.agent import ClosedLoopTumorAgent
from tumor_orchestration.schema import load_config
from tumor_orchestration.training import (
    configure_logging,
    create_optimizer,
    device_for_local_rank,
    finalize_distributed,
    initialize_distributed,
    set_seed,
)

LOGGER = logging.getLogger("tumor_orchestration.commands")


def train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tumor-train")
    parser.add_argument("--config", type=Path, default=Path("configurations/main.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/main"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-level", default="INFO")
    return parser


def evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tumor-evaluate")
    parser.add_argument("--config", type=Path, default=Path("configurations/main.yaml"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--log-level", default="INFO")
    return parser


def model_summary(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    buffers = sum(buffer.numel() for buffer in model.buffers())
    return {"parameters": total, "trainable": trainable, "buffers": buffers}


def train_main(arguments: Sequence[str] | None = None) -> None:
    options = train_parser().parse_args(arguments)
    configure_logging(getattr(logging, options.log_level.upper()))
    process_rank, process_world, local_rank = initialize_distributed()
    config = load_config(options.config)
    if process_world != config.training.world_size:
        raise RuntimeError(
            f"configuration requires world size {config.training.world_size}, "
            f"received {process_world}"
        )
    set_seed(config.seed + process_rank)
    device = device_for_local_rank(local_rank)
    model = ClosedLoopTumorAgent(config).to(device)
    optimizer = create_optimizer(model, config)
    summary = model_summary(model)
    if process_rank == 0:
        options.output.mkdir(parents=True, exist_ok=True)
        (options.output / "model_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    LOGGER.info(
        "initialized rank=%d world=%d device=%s parameters=%d data=%s optimizer=%s",
        process_rank,
        process_world,
        device,
        summary["parameters"],
        options.data,
        optimizer.__class__.__name__,
    )
    finalize_distributed()


def evaluate_main(arguments: Sequence[str] | None = None) -> None:
    options = evaluate_parser().parse_args(arguments)
    configure_logging(getattr(logging, options.log_level.upper()))
    config = load_config(options.config)
    payload = torch.load(options.weights, map_location="cpu")
    model = ClosedLoopTumorAgent(config)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    predictions = json.loads(options.predictions.read_text(encoding="utf-8"))
    output = {
        "model": model_summary(model),
        "records": len(predictions),
        "bootstrap_iterations": config.evaluation.bootstrap_iterations,
        "folds": config.evaluation.folds,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("evaluation metadata written to %s", options.output)


def main(arguments: Sequence[str] | None = None) -> None:
    train_main(arguments)


if __name__ == "__main__":
    main()
