# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Evaluate command for openbench-cli."""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import typer
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.table import Table

from openbench.dataset import DatasetRegistry
from openbench.metric import MetricOptions
from openbench.pipeline import PipelineRegistry
from openbench.runner import BenchmarkConfig, BenchmarkResult, BenchmarkRunner, WandbConfig

from ..command_utils import (
    get_datasets_help_text,
    get_metrics_help_text,
    get_pipelines_help_text,
    parse_pipeline_config_overrides,
    validate_dataset_name,
    validate_pipeline_dataset_compatibility,
    validate_pipeline_metrics_compatibility,
    validate_pipeline_name,
)


PipelineConfigOptions = dict[str, dict[str, Any] | dict[str, dict[str, Any]]]


def parse_dataset_shard(spec: str) -> tuple[int, int]:
    """Parse a ``INDEX/TOTAL`` shard spec into 0-based (index, total)."""
    index_str, sep, total_str = spec.partition("/")
    if not sep:
        raise typer.BadParameter(f"Expected INDEX/TOTAL, got {spec!r}", param_hint="--dataset-shard")
    try:
        index, total = int(index_str), int(total_str)
    except ValueError:
        raise typer.BadParameter(f"INDEX and TOTAL must be integers, got {spec!r}", param_hint="--dataset-shard")
    if total < 1 or not 0 <= index < total:
        raise typer.BadParameter(f"Need 0 <= INDEX < TOTAL and TOTAL >= 1, got {spec!r}", param_hint="--dataset-shard")
    return index, total


class EvaluationConfig(BaseModel):
    benchmark_config: BenchmarkConfig = Field(..., description="The benchmark config to use for evaluation")
    pipeline_config: dict[str, dict[str, Any]] = Field(
        ..., description="The pipeline config to use for evaluation where the key is the pipeline name"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_pipeline_config(cls, v: dict[str, Any]) -> dict[str, Any]:
        # Helper to support previous evaluation config .yamls used in previous versions of openbench
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
    try:
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
    except Exception as e:
        typer.echo(f"❌ Failed to load evaluation config from {evaluation_config_path}: {e}", err=True)
        sys.exit(1)


def run_config_file_mode(
    evaluation_config_path: Path,
    evaluation_config_overrides: list[str] | None,
    verbose: bool,
) -> BenchmarkResult:
    """Run evaluation using a config file."""
    if verbose:
        typer.echo("🚀 Starting evaluation with config file...")
        typer.echo(f"✅ Config file: {evaluation_config_path}")
        if evaluation_config_overrides:
            typer.echo(f"✅ Overrides: {evaluation_config_overrides}")

    try:
        config = load_evaluation_config(evaluation_config_path, evaluation_config_overrides)
        benchmark_config = config.benchmark_config

        if not config.pipeline_config:
            typer.echo("❌ No pipeline configuration found in evaluation config", err=True)
            sys.exit(1)

        pipeline_class_name, pipeline_config = list(config.pipeline_config.items())[0]

        # Create pipeline
        typer.echo(f"🔧 Creating pipeline: {pipeline_class_name}")
        pipeline = PipelineRegistry.create_pipeline(name=pipeline_class_name, config=pipeline_config)
        benchmark_runner = BenchmarkRunner(config=benchmark_config, pipelines=[pipeline])

        if verbose:
            typer.echo(f"✅ Pipeline: {pipeline_class_name}")
            typer.echo(f"✅ Datasets: {list(benchmark_config.datasets.keys())}")
            typer.echo(f"✅ Metrics: {list(benchmark_config.metrics.keys())}")
            typer.echo(f"✅ WandB: {'enabled' if benchmark_config.wandb_config.is_active else 'disabled'}")

        typer.echo("🚀 Starting evaluation...")
        result = benchmark_runner.run()

        if result:
            typer.echo("✅ Evaluation completed successfully!")
            if verbose:
                typer.echo(
                    f"📊 Results saved to: {result.output_dir if hasattr(result, 'output_dir') else 'current directory'}"
                )
        else:
            typer.echo("⚠️  Evaluation completed but no results were returned", err=True)

        return result

    except Exception as e:
        typer.echo(f"❌ Evaluation failed: {e}", err=True)
        if verbose:
            import traceback

            typer.echo(f"📋 Full traceback:\n{traceback.format_exc()}", err=True)
        sys.exit(1)


def run_alias_mode(
    pipeline_name: str,
    dataset_name: str,
    metrics: list[MetricOptions],
    use_wandb: bool,
    wandb_project: str,
    wandb_run_name: str | None,
    wandb_tags: list[str] | None,
    use_keywords: bool | None,
    force_language: bool,
    pipeline_config: list[str] | None,
    verbose: bool,
    hf_results_repo: str | None = None,
    hf_results_flush_every: int = 100,
    hf_results_chunk_tag: str | None = None,
    dataset_shard: str | None = None,
    skip_completed_in: str | None = None,
    metric_config: list[str] | None = None,
) -> BenchmarkResult:
    """Run evaluation using pipeline and dataset aliases."""
    try:
        # Validate cross-parameter compatibility
        typer.echo("🔍 Validating configuration...")
        validate_pipeline_dataset_compatibility(pipeline_name, dataset_name)
        validate_pipeline_metrics_compatibility(pipeline_name, metrics)

        if verbose:
            dataset_info = DatasetRegistry.get_alias_info(dataset_name)
            typer.echo(f"✅ Pipeline: {pipeline_name}")
            typer.echo(f"✅ Dataset: {dataset_name} ({dataset_info.config.dataset_id})")
            typer.echo(f"✅ Metrics: {[m.value for m in metrics]}")
            typer.echo(f"✅ WandB: {'enabled' if use_wandb else 'disabled'}")

        ######### Build Pipeline #########
        typer.echo(f"🔧 Creating pipeline: {pipeline_name}")

        # Handle use_keywords override
        pipeline_config_override = {}
        if use_keywords is not None:
            pipeline_config_override["use_keywords"] = use_keywords
            if verbose:
                typer.echo(f"✅ Keywords: {'enabled' if use_keywords else 'disabled'} (override)")

        # Handle force_language override
        if force_language:
            pipeline_config_override["force_language"] = force_language
            if verbose:
                typer.echo("✅ Force language: enabled")

        # Handle generic pipeline config overrides (key=value pairs).
        # Values are kept as strings; the pipeline's Pydantic config
        # coerces them to int/float/bool/etc when instantiated.
        for key, value in parse_pipeline_config_overrides(pipeline_config).items():
            pipeline_config_override[key] = value
            if verbose:
                typer.echo(f"Config override: {key}={value}")

        pipeline = PipelineRegistry.create_pipeline(pipeline_name, config=pipeline_config_override)

        ######### Build Benchmark Config #########
        typer.echo(f"📊 Loading dataset: {dataset_name}")
        dataset_config = DatasetRegistry.get_alias_config(dataset_name)

        # Row selection for split-up sweeps: keep one shard, and/or drop samples
        # that already have results elsewhere.
        dataset_overrides: dict[str, Any] = {}
        if dataset_shard:
            shard_index, num_shards = parse_dataset_shard(dataset_shard)
            dataset_overrides.update(num_shards=num_shards, shard_index=shard_index)
            typer.echo(f"🔀 Dataset shard {shard_index} of {num_shards} (interleaved)")
        if skip_completed_in:
            from openbench.runner.speech_generation_sink import completed_sample_ids

            repos = [repo for repo in (r.strip() for r in skip_completed_in.split(",")) if repo]
            completed = completed_sample_ids(repos)
            typer.echo(f"⏭️  Skipping {len(completed)} samples already scored in {', '.join(repos)}")
            if completed:
                dataset_overrides["exclude_sample_ids"] = frozenset(completed)
        if dataset_overrides:
            dataset_config = dataset_config.model_copy(update=dataset_overrides)

        wandb_config = WandbConfig(
            project_name=wandb_project,
            run_name=wandb_run_name,
            tags=wandb_tags,
            is_active=use_wandb,
        )

        # Metric constructor kwargs (`-mc metric.key=value`), e.g. the SIM
        # speaker-encoder checkpoint. Values stay strings; metric __init__
        # signatures accept them directly or coerce as needed.
        metric_kwargs: dict[MetricOptions, dict[str, Any]] = {metric: {} for metric in metrics}
        for item in metric_config or []:
            key, sep, value = item.partition("=")
            metric_name, dot, field = key.partition(".")
            if not sep or not dot or not field or not value:
                raise typer.BadParameter(f"Expected metric.key=value, got {item!r}", param_hint="--metric-config")
            try:
                metric_option = MetricOptions(metric_name)
            except ValueError:
                raise typer.BadParameter(f"Unknown metric {metric_name!r} in {item!r}", param_hint="--metric-config")
            if metric_option not in metric_kwargs:
                raise typer.BadParameter(
                    f"Metric {metric_name!r} is not among --metrics {[m.value for m in metrics]}",
                    param_hint="--metric-config",
                )
            metric_kwargs[metric_option][field] = value
            if verbose:
                typer.echo(f"Metric config override: {metric_name}.{field}={value}")

        benchmark_config = BenchmarkConfig(
            wandb_config=wandb_config,
            datasets={dataset_name: dataset_config},
            metrics=metric_kwargs,
            hf_results_repo=hf_results_repo,
            hf_results_flush_every=hf_results_flush_every,
            hf_results_chunk_tag=hf_results_chunk_tag,
        )

        # Create runner
        benchmark_runner = BenchmarkRunner(config=benchmark_config, pipelines=[pipeline])

        typer.echo("🚀 Starting evaluation...")
        result = benchmark_runner.run()

        if result:
            typer.echo("✅ Evaluation completed successfully!")
            if verbose:
                typer.echo(
                    f"📊 Results saved to: {result.output_dir if hasattr(result, 'output_dir') else 'current directory'}"
                )
        else:
            typer.echo("⚠️  Evaluation completed but no results were returned", err=True)

        return result

    except Exception as e:
        typer.echo(f"❌ Evaluation failed: {e}", err=True)
        if verbose:
            import traceback

            typer.echo(f"📋 Full traceback:\n{traceback.format_exc()}", err=True)
        sys.exit(1)


BASE_OUTPUT_DIR = Path("outputs")


def get_output_dir() -> Path:
    """Get the output directory for the evaluation."""
    now_utc = datetime.now()
    date_str = now_utc.strftime("%Y-%m-%d")
    time_str = now_utc.strftime("%H-%M-%S")

    output_dir = BASE_OUTPUT_DIR / date_str / time_str
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def display_result(result: BenchmarkResult) -> None:
    """Display global benchmark result for every metric in a pretty rich table."""
    table = Table(title="Benchmark Result")
    table.add_column("Dataset", justify="center", style="cyan", no_wrap=True)
    table.add_column("Pipeline", justify="center", style="cyan", no_wrap=True)
    table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
    table.add_column("Value", justify="center", style="cyan", no_wrap=True)

    global_results = result.global_results
    for global_result in global_results:
        # For streaming pipelines we can have metrics that are not compatible and return N/A values
        if global_result.global_result is None:
            continue
        row = [
            global_result.dataset_name,
            global_result.pipeline_name,
            global_result.metric_name,
            f"{global_result.global_result:.4f}",
        ]
        table.add_row(*row)
    console = Console()
    console.print(table)


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
        "openbench-eval", "--wandb-project", "-wp", help="W&B project to use for evaluation"
    ),
    wandb_run_name: str | None = typer.Option(
        None, "--wandb-run-name", "-wr", help="W&B run name to use for evaluation"
    ),
    wandb_tags: list[str] | None = typer.Option(None, "--wandb-tags", "-wt", help="W&B tags to use for evaluation"),
    use_keywords: bool | None = typer.Option(
        None,
        "--use-keywords",
        help="Enable keyword boosting for compatible pipelines (overrides default config)",
    ),
    force_language: bool = typer.Option(
        False,
        "--force-language",
        help="Force language hinting for compatible pipelines",
    ),
    pipeline_config: list[str] | None = typer.Option(
        None,
        "--pipeline-config",
        "-pc",
        help=(
            "Override one or more pipeline config fields as key=value pairs. "
            "The value is parsed as a string; Pydantic coerces it to the field's "
            "declared type when the pipeline is instantiated, so ints/floats/bools "
            "all just work. Repeat the flag for multiple overrides. Examples: "
            "`-pc speaker=serena`, `-pc seed=42 -pc temperature=0.7`, "
            "`-pc force_language=true`."
        ),
    ),
    metric_config: list[str] | None = typer.Option(
        None,
        "--metric-config",
        "-mc",
        help=(
            "Pass one or more metric constructor kwargs as metric.key=value pairs "
            "(alias mode only). Repeat the flag for multiple overrides. Example: "
            "`-mc sim.checkpoint=/path/to/wavlm_large_finetune.pth` "
            "(the SIM metric also accepts hf://owner/repo/filename)."
        ),
    ),
    hf_results_repo: str | None = typer.Option(
        None,
        "--hf-results-repo",
        help=(
            "Push per-sample speech-generation results (reference/generated audio + SIM/WER) "
            "incrementally to this Hugging Face dataset repo as append-only parquet shards. "
            "Example: argmaxinc/openbench-speech-generation-benchmark"
        ),
    ),
    hf_results_flush_every: int = typer.Option(
        100,
        "--hf-results-flush-every",
        help="Flush buffered per-sample results to the HF repo every N samples (used with --hf-results-repo).",
    ),
    hf_results_chunk_tag: str | None = typer.Option(
        None,
        "--hf-results-chunk-tag",
        help=(
            "Suffix for the uploaded parquet shard filenames, e.g. `--hf-results-chunk-tag s3`. "
            "Required when several runs push to the same --hf-results-repo concurrently: each "
            "picks its starting chunk index independently, so without distinct tags they would "
            "overwrite each other's shards."
        ),
    ),
    dataset_shard: str | None = typer.Option(
        None,
        "--dataset-shard",
        help=(
            "Evaluate only one interleaved shard of the dataset, given as INDEX/TOTAL with a "
            "0-based index, e.g. `--dataset-shard 3/10`. Lets one sweep run as several independent "
            "jobs, each short enough to finish inside a CI job's time limit. Shard membership "
            "depends only on INDEX/TOTAL, so a single shard can be retried on its own. Alias mode only."
        ),
    ),
    skip_completed_in: str | None = typer.Option(
        None,
        "--skip-completed-in",
        help=(
            "Comma-separated HF results repos to resume past: every sample already scored there is "
            "dropped before evaluation. Pass the current --hf-results-repo plus any older repo "
            "holding partial results. Alias mode only."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Run evaluation benchmarks.

    This command supports two modes:

    1. **Config file mode**: Provide --evaluation-config for full control

    2. **Alias mode**: Use pre-configured pipeline/dataset aliases with --pipeline, --dataset, --metrics

    Examples:

        # Config file mode

        openbench-cli evaluate --evaluation-config config/my_eval.yaml

        # Alias mode - evaluate pyannote pipeline on voxconverse dataset with DER and JER metrics

        openbench-cli evaluate --pipeline pyannote --dataset voxconverse --metrics der jer
    """
    # Validate required parameters
    if evaluation_config_path is None and (pipeline_name is None or dataset_name is None or metrics is None):
        raise typer.BadParameter(
            "Must provide either --evaluation-config or --pipeline, --dataset, and --metrics\n\n"
            "Examples:\n"
            "  openbench-cli evaluate --evaluation-config config/my_eval.yaml\n"
            "  openbench-cli evaluate --pipeline pyannote --dataset voxconverse --metrics der jer"
        )

    # Get output dir
    output_dir = get_output_dir()
    # Tell user which output dir is being used for the run
    typer.echo(f"📁 Output directory: {output_dir}")

    # Store original working directory
    original_cwd = os.getcwd()

    # Get absolute path for evaluation config before changing working directory
    if evaluation_config_path is not None:
        evaluation_config_path = evaluation_config_path.absolute()

    try:
        # Set output_dir as working dir
        os.chdir(output_dir)

        # Validate mutually exclusive modes
        if evaluation_config_path is not None:
            alias_only = {
                "--dataset-shard": dataset_shard,
                "--skip-completed-in": skip_completed_in,
                "--hf-results-chunk-tag": hf_results_chunk_tag,
            }
            unsupported = [flag for flag, value in alias_only.items() if value]
            if unsupported:
                raise typer.BadParameter(
                    f"{', '.join(unsupported)} only apply in alias mode; set the equivalent fields in "
                    "the evaluation config instead."
                )
            typer.echo("🔧 Running with config file mode")
            result = run_config_file_mode(evaluation_config_path, evaluation_config_overrides, verbose)
        else:
            typer.echo("🔧 Running with alias mode")
            result = run_alias_mode(
                pipeline_name=pipeline_name,
                dataset_name=dataset_name,
                metrics=metrics,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_run_name=wandb_run_name,
                wandb_tags=wandb_tags,
                use_keywords=use_keywords,
                force_language=force_language,
                pipeline_config=pipeline_config,
                hf_results_repo=hf_results_repo,
                hf_results_flush_every=hf_results_flush_every,
                hf_results_chunk_tag=hf_results_chunk_tag,
                dataset_shard=dataset_shard,
                skip_completed_in=skip_completed_in,
                metric_config=metric_config,
                verbose=verbose,
            )
        display_result(result)

    finally:
        # Restore original working directory
        os.chdir(original_cwd)
