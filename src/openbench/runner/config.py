# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import Any

from pydantic import BaseModel, Field

from ..dataset import DatasetConfig
from ..metric import MetricOptions


class WandbConfig(BaseModel):
    project_name: str = Field(..., description="The name of the project for wandb")
    run_name: str | None = Field(None, description="The name of the run for wandb")
    is_active: bool = Field(True, description="Whether to log to wandb")
    tags: list[str] | None = Field(None, description="Tags for the run")

    @property
    def wandb_mode(self) -> str:
        return "disabled" if not self.is_active else "online"


class BenchmarkConfig(BaseModel):
    wandb_config: WandbConfig = Field(..., description="The wandb config")
    # a dictionary mapping tasks types to metric options and their initialization kwargs
    metrics: dict[MetricOptions, dict[str, Any]] = Field(
        ..., description="The metrics that will be used for each task"
    )
    datasets: dict[str, DatasetConfig] = Field(..., description="Datasets to evaluate")
    hf_results_repo: str | None = Field(
        None,
        description=(
            "If set, per-sample speech-generation results (audio + SIM/WER) are pushed "
            "incrementally to this Hugging Face dataset repo as append-only parquet shards."
        ),
    )
    hf_results_flush_every: int = Field(
        100, description="Flush buffered per-sample results to the HF repo every N samples."
    )
    hf_results_chunk_tag: str | None = Field(
        None,
        description=(
            "Suffix appended to the uploaded parquet shard names. Required when several runs "
            "(e.g. dataset shards) push to one repo at the same time, since each picks its "
            "starting chunk index independently and identical names would overwrite."
        ),
    )
    continue_on_sample_error: bool = Field(
        True,
        description=(
            "If True (default), a sample that fails during processing is logged and skipped so the "
            "rest of the dataset still runs; metrics are computed over the successful samples. "
            "Set False to abort the whole run on the first sample failure (sequential mode only)."
        ),
    )

    class Config:
        arbitrary_types_allowed = True

    def get_wandb_config_to_log(self) -> dict[str, Any]:
        wandb_config: dict[str, Any] = self.model_dump()
        # Convert `metrics` that use enums to their respective values
        wandb_config["metrics"] = {metric.value: kwargs for metric, kwargs in wandb_config["metrics"].items()}
        return wandb_config
