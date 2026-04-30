# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Common utilities for CLI commands."""

import typer

from openbench.dataset import DatasetRegistry
from openbench.metric import MetricOptions, MetricRegistry
from openbench.pipeline import PipelineRegistry
from openbench.types import PipelineType


def get_available_pipelines() -> list[str]:
    """Get list of available pipeline names."""
    return PipelineRegistry.list_pipelines()


def parse_pipeline_config_overrides(items: list[str] | None) -> dict[str, str]:
    """Parse a list of `key=value` strings into a config-override dict.

    Values are kept as raw strings. Pydantic performs the actual type
    coercion when the pipeline config is built — so e.g. `seed=10`,
    `temperature=0.9`, and `force_language=true` all work as expected
    against a `SpeechGenerationConfig` (or any other Pydantic config),
    because Pydantic's lax mode parses ints, floats, and the standard
    boolean spellings ("true"/"false"/"yes"/"no"/"1"/"0", case-insensitive)
    from strings. Quote arguments that contain spaces.

    Examples:
        --pipeline-config speaker=serena
        --pipeline-config seed=42 -pc temperature=0.7
        --pipeline-config force_language=true

    Args:
        items: The raw values typer collected for `--pipeline-config`,
            or `None` if the flag wasn't passed.

    Returns:
        Dict mapping config keys to (string) values. Empty if `items`
        is `None` or empty.

    Raises:
        typer.BadParameter: If any item is missing the `=` separator,
            or has an empty key.
    """
    overrides: dict[str, str] = {}
    if not items:
        return overrides

    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid --pipeline-config format: '{item}'. Expected key=value (e.g. seed=42)")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"Invalid --pipeline-config format: '{item}'. Empty key before '='.")
        overrides[key] = value
    return overrides


def get_available_datasets() -> list[str]:
    """Get list of available dataset aliases."""
    return DatasetRegistry.list_aliases()


def get_metrics_help_text() -> str:
    """Generate help text for metrics organized by pipeline type."""
    help_text = "Available metrics by pipeline type:\n\n"

    # Group metrics by pipeline type
    metrics_by_pipeline = {}
    for pipeline_type in PipelineType:
        available_metrics = MetricRegistry.get_available_metrics(pipeline_type)
        if available_metrics:
            metrics_by_pipeline[pipeline_type.name] = [metric.value for metric in available_metrics]

    for pipeline_type, metrics in metrics_by_pipeline.items():
        help_text += f"  {pipeline_type}:\n\n"
        for metric in sorted(metrics):
            help_text += f"- {metric}\n\n"

    return help_text


def get_pipelines_help_text() -> str:
    """Generate help text for pipeline aliases organized by type."""
    help_text = "Available pipeline aliases by type:\n\n"

    # Group pipeline aliases by type
    pipelines_by_type = {}
    for pipeline_name in get_available_pipelines():
        # Only include aliases, not class names
        if PipelineRegistry.is_alias(pipeline_name):
            pipeline_type = PipelineRegistry.get_pipeline_type(pipeline_name)
            pipeline_type_name = pipeline_type.name
            if pipeline_type_name not in pipelines_by_type:
                pipelines_by_type[pipeline_type_name] = []
            pipelines_by_type[pipeline_type_name].append(pipeline_name)

    for pipeline_type, pipelines in pipelines_by_type.items():
        help_text += f"  {pipeline_type}:\n\n"
        for pipeline in sorted(pipelines):
            help_text += f"- {pipeline}\n\n"

    return help_text


def get_datasets_help_text() -> str:
    """Generate help text for datasets organized by pipeline type."""
    help_text = "Available datasets by pipeline type:\n\n"

    # Group datasets by pipeline type
    datasets_by_pipeline = {}
    for alias in get_available_datasets():
        info = DatasetRegistry.get_alias_info(alias)
        for pipeline_type in info.supported_pipeline_types:
            pipeline_type_name = pipeline_type.name
            if pipeline_type_name not in datasets_by_pipeline:
                datasets_by_pipeline[pipeline_type_name] = []
            datasets_by_pipeline[pipeline_type_name].append(alias)

    for pipeline_type, datasets in datasets_by_pipeline.items():
        help_text += f"  {pipeline_type}:\n\n"
        for dataset in sorted(datasets):
            help_text += f"- {dataset}\n\n"

    return help_text


def validate_pipeline_name(pipeline_name: str | None) -> str:
    """Validate that the pipeline name exists in the registry."""
    if pipeline_name is None:
        return None

    try:
        PipelineRegistry.get_pipeline_class(pipeline_name)
    except ValueError:
        available = ", ".join(get_available_pipelines())
        raise typer.BadParameter(f"Pipeline '{pipeline_name}' not found. Available pipelines: {available}")
    return pipeline_name


def validate_dataset_name(dataset_name: str | None) -> str:
    """Validate that the dataset alias exists in the registry."""
    if dataset_name is None:
        return None

    if not DatasetRegistry.has_alias(dataset_name):
        available = ", ".join(get_available_datasets())
        raise typer.BadParameter(f"Dataset '{dataset_name}' not found. Available datasets: {available}")
    return dataset_name


def validate_pipeline_dataset_compatibility(pipeline_name: str, dataset_name: str) -> tuple[str, str]:
    """Validate that the pipeline is compatible with the dataset."""
    pipeline_type = PipelineRegistry.get_pipeline_type(pipeline_name)
    dataset_info = DatasetRegistry.get_alias_info(dataset_name)

    if pipeline_type not in dataset_info.supported_pipeline_types:
        supported_types = [t.name for t in dataset_info.supported_pipeline_types]
        raise typer.BadParameter(
            f"Pipeline '{pipeline_name}' ({pipeline_type.name}) is not compatible with dataset '{dataset_name}'. "
            f"Dataset supports: {', '.join(supported_types)}"
        )
    return pipeline_name, dataset_name


def validate_pipeline_metrics_compatibility(
    pipeline_name: str, metrics: list[MetricOptions]
) -> tuple[str, list[MetricOptions]]:
    """Validate that the metrics are available for the pipeline type."""
    pipeline_type = PipelineRegistry.get_pipeline_type(pipeline_name)
    available_metrics = MetricRegistry.get_available_metrics(pipeline_type)
    incompatible_metrics = [m for m in metrics if m not in available_metrics]

    if incompatible_metrics:
        available = ", ".join([m.value for m in available_metrics])
        incompatible = ", ".join([m.value for m in incompatible_metrics])
        raise typer.BadParameter(
            f"Metrics {incompatible} are not available for pipeline type '{pipeline_type.name}'. "
            f"Available metrics for this pipeline type: {available}"
        )
    return pipeline_name, metrics
