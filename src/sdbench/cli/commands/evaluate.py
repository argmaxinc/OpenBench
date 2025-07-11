# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Evaluate command for sdbench-cli."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import typer
from pydantic import BaseModel, Field, model_validator

from sdbench.dataset import DatasetRegistry
from sdbench.metric import MetricOptions
from sdbench.pipeline import PipelineRegistry
from sdbench.runner import BenchmarkConfig, BenchmarkRunner, WandbConfig

from ..command_utils import (
    get_datasets_help_text,
    get_metrics_help_text,
    get_pipelines_help_text,
    validate_dataset_name,
    validate_pipeline_dataset_compatibility,
    validate_pipeline_metrics_compatibility,
    validate_pipeline_name,
)


PipelineConfigOptions = dict[str, dict[str, Any] | dict[str, dict[str, Any]]]


class EvaluationConfig(BaseModel):
    benchmark_config: BenchmarkConfig = Field(..., description="The benchmark config to use for evaluation")
    pipeline_config: dict[str, dict[str, Any]] = Field(
        ..., description="The pipeline config to use for evaluation where the key is the pipeline name"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_pipeline_config(cls, v: dict[str, Any]) -> dict[str, Any]:
        # Helper to support previous evaluation config .yamls used in previous versions of sdbench
        # the input for a pipeline_config could be:
        # - a dict where the key is the pipeline class name and the value is the parameters for the configuration
        # - a dict where the key is the pipeline class name and the value is a dict with key `config` and value is the parameters for the configuration
        # For full backward compatibility we should support the case where `pipeline_configs` is passed instead of `pipeline_config`
        # in the end we should still populate the `pipeline_config` field.

        if isinstance(v, dict):
            # Handle the case where pipeline_configs is passed instead of pipeline_config
            if "pipeline_configs" in v and "pipeline_config" not in v:
                v["pipeline_config"] = v.pop("pipeline_configs")

            # Handle pipeline_config normalization
            if "pipeline_config" in v:
                pipeline_config = v["pipeline_config"]
                if isinstance(pipeline_config, dict):
                    normalized_config = {}
                    for pipeline_name, config_value in pipeline_config.items():
                        if isinstance(config_value, dict) and "config" in config_value:
                            # Case: {"pipeline_name": {"config": {...}}}
                            normalized_config[pipeline_name] = config_value["config"]
                        else:
                            # Case: {"pipeline_name": {...}}
                            normalized_config[pipeline_name] = config_value
                    v["pipeline_config"] = normalized_config

        return v

    class Config:
        arbitrary_types_allowed = True


def load_evaluation_config(
    evaluation_config_path: Path, evaluation_config_overrides: list[str] | None
) -> EvaluationConfig:
    """Load an evaluation config from a file.
    This function uses Hydra to load the evaluation config from a file.
    It then returns an `EvaluationConfig` object.

    Args:
        evaluation_config_path: The path to the evaluation config file.
        evaluation_config_overrides: The overrides to apply to the evaluation config.
    """
    # Current dir of this file
    base_dir = Path(__file__).parent
    # Get dir for config
    config_dir = evaluation_config_path.absolute().parent
    # Get config name
    config_name = evaluation_config_path.stem

    # Get the relative path from the base dir to the config dir
    relative_config_dir = os.path.relpath(config_dir, start=base_dir)

    # Initialize Hydra with the config path
    with hydra.initialize(config_path=relative_config_dir):
        config = hydra.compose(
            config_name=config_name,
            overrides=evaluation_config_overrides,
        )

        return EvaluationConfig(**config)


def run_config_file_mode(
    evaluation_config_path: Path,
    evaluation_config_overrides: list[str] | None,
    verbose: bool,
) -> None:
    """Run evaluation using a config file."""
    if verbose:
        typer.echo("🚀 Starting evaluation with config file...")
        typer.echo(f"✅ Config file: {evaluation_config_path}")
        if evaluation_config_overrides:
            typer.echo(f"✅ Overrides: {evaluation_config_overrides}")

    config = load_evaluation_config(evaluation_config_path, evaluation_config_overrides)
    benchmark_config = config.benchmark_config
    pipeline_class_name, pipeline_config = list(config.pipeline_config.items())[0]

    # Create pipeline
    pipeline = PipelineRegistry.create_pipeline(name=pipeline_class_name, config=pipeline_config)
    benchmark_runner = BenchmarkRunner(config=benchmark_config, pipelines=[pipeline])

    if verbose:
        typer.echo(f"✅ Pipeline: {pipeline_class_name}")
        typer.echo(f"✅ Dataset: {benchmark_config.datasets.keys()}")
        typer.echo(f"✅ Metrics: {list(benchmark_config.metrics.keys())}")
        typer.echo(f"✅ WandB: {'enabled' if benchmark_config.wandb_config.is_active else 'disabled'}")

    typer.echo("🚀 Starting evaluation...")
    _ = benchmark_runner.run()
    typer.echo("✅ Evaluation completed successfully!")


def run_alias_mode(
    pipeline_name: str,
    dataset_name: str,
    metrics: list[MetricOptions],
    use_wandb: bool,
    wandb_project: str,
    wandb_run_name: str | None,
    wandb_tags: list[str] | None,
    verbose: bool,
) -> None:
    """Run evaluation using pipeline and dataset aliases."""
    # Validate cross-parameter compatibility
    validate_pipeline_dataset_compatibility(pipeline_name, dataset_name)
    validate_pipeline_metrics_compatibility(pipeline_name, metrics)

    if verbose:
        dataset_info = DatasetRegistry.get_alias_info(dataset_name)
        typer.echo(f"✅ Pipeline: {pipeline_name}")
        typer.echo(f"✅ Dataset: {dataset_name} ({dataset_info.config.dataset_id})")
        typer.echo(f"✅ Metrics: {[m.value for m in metrics]}")
        typer.echo(f"✅ WandB: {'enabled' if use_wandb else 'disabled'}")

    ######### Build Pipeline #########
    pipeline = PipelineRegistry.create_pipeline(pipeline_name)

    ######### Build Benchmark Config #########
    dataset_config = DatasetRegistry.get_alias_config(dataset_name)

    wandb_config = WandbConfig(
        project_name=wandb_project,
        run_name=wandb_run_name,
        tags=wandb_tags,
        is_active=use_wandb,
    )

    benchmark_config = BenchmarkConfig(
        wandb_config=wandb_config, datasets={dataset_name: dataset_config}, metrics={metric: {} for metric in metrics}
    )

    # Create runner
    benchmark_runner = BenchmarkRunner(config=benchmark_config, pipelines=[pipeline])

    typer.echo("🚀 Starting evaluation...")
    _ = benchmark_runner.run()
    typer.echo("✅ Evaluation completed successfully!")


BASE_OUTPUT_DIR = Path("outputs")


def get_output_dir() -> Path:
    """Get the output directory for the evaluation."""
    now_utc = datetime.now()
    date_str = now_utc.strftime("%Y-%m-%d")
    time_str = now_utc.strftime("%H-%M-%S")

    output_dir = BASE_OUTPUT_DIR / date_str / time_str
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def evaluate(
    evaluation_config_path: Path | None = typer.Option(
        None,
        "--evaluation-config",
        "-ec",
        help=(
            "Path to an evaluation config file for full control over evaluation settings. "
            "When provided, this overrides all other CLI options. "
            "The config should define datasets, pipelines, metrics, and W&B settings."
        ),
    ),
    evaluation_config_overrides: list[str] | None = typer.Option(
        None, "--evaluation-config-overrides", "-eov", help="Hydra overrides to apply to the evaluation config file"
    ),
    pipeline_name: str | None = typer.Option(
        None,
        "--pipeline",
        "-p",
        help=f"The name of the registered pipeline to use for evaluation\n\n{get_pipelines_help_text()}",
        callback=validate_pipeline_name,
    ),
    dataset_name: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help=f"The alias of the registered dataset to use for evaluation\n\n{get_datasets_help_text()}",
        callback=validate_dataset_name,
    ),
    # Metrics don't need validation. Typer validates already since we use the MetricOptions enum
    metrics: list[MetricOptions] | None = typer.Option(
        None,
        "--metrics",
        "-m",
        help=f"The metrics to use for evaluation\n\n{get_metrics_help_text()}",
    ),
    ######## WandB arguments ########
    use_wandb: bool = typer.Option(False, "--use-wandb", "-w", help="Use W&B for evaluation"),
    wandb_project: str = typer.Option(
        "sdbench-eval", "--wandb-project", "-wp", help="W&B project to use for evaluation"
    ),
    wandb_run_name: str | None = typer.Option(
        None, "--wandb-run-name", "-wr", help="W&B run name to use for evaluation"
    ),
    wandb_tags: list[str] | None = typer.Option(None, "--wandb-tags", "-wt", help="W&B tags to use for evaluation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Run evaluation benchmarks.

    This command supports two modes:

    1. **Config file mode**: Provide --evaluation-config for full control

    2. **Alias mode**: Use pre-configured pipeline/dataset aliases with --pipeline, --dataset, --metrics

    Examples:

        # Config file mode

        sdbench-cli evaluate --evaluation-config config/my_eval.yaml

        # Alias mode - evaluate pyannote pipeline on voxconverse dataset with DER and JER metrics

        sdbench-cli evaluate --pipeline pyannote --dataset voxconverse --metrics der jer
    """
    if evaluation_config_path is None and (pipeline_name is None or dataset_name is None or metrics is None):
        raise typer.BadParameter("Must provide either --evaluation-config or --pipeline, --dataset, and --metrics")

    # Get output dir
    output_dir = get_output_dir()
    # Tell user which output dir is being used for the run
    typer.echo(f"🏃 output directory: {output_dir}")
    # Set output_dir as working dir
    os.chdir(output_dir)

    # Validate mutually exclusive modes
    if evaluation_config_path is not None:
        typer.echo("Running with config file mode")
        run_config_file_mode(evaluation_config_path, evaluation_config_overrides, verbose)
        return

    typer.echo("Running with alias mode")
    run_alias_mode(
        pipeline_name=pipeline_name,
        dataset_name=dataset_name,
        metrics=metrics,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_tags=wandb_tags,
        verbose=verbose,
    )
