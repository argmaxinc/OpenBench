# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from dataclasses import dataclass
from typing import ClassVar

from ..types import PipelineType
from .dataset_base import BaseDataset, DatasetConfig
from .dataset_diarization import DiarizationDataset
from .dataset_orchestration import OrchestrationDataset
from .dataset_streaming_transcription import StreamingDataset
from .dataset_transcription import TranscriptionDataset


@dataclass
class DatasetAliasInfo:
    """Information about a dataset alias."""

    config: DatasetConfig
    supported_pipeline_types: set[PipelineType]
    description: str = ""


class DatasetRegistry:
    """Registry for datasets by pipeline type."""

    _datasets: ClassVar[dict[PipelineType, type[BaseDataset]]] = {}
    _aliases: ClassVar[dict[str, DatasetAliasInfo]] = {}

    @classmethod
    def register(cls, pipeline_type: PipelineType, dataset_class: type[BaseDataset]) -> None:
        """Register a dataset class for a specific pipeline type."""
        cls._datasets[pipeline_type] = dataset_class

    @classmethod
    def register_alias(
        cls, alias: str, config: DatasetConfig, supported_pipeline_types: set[PipelineType], description: str = ""
    ) -> None:
        """Register a dataset alias with its config and supported pipeline types."""
        cls._aliases[alias] = DatasetAliasInfo(
            config=config, supported_pipeline_types=supported_pipeline_types, description=description
        )

    @classmethod
    def get_dataset_for_pipeline(cls, pipeline_type: PipelineType, config: DatasetConfig) -> BaseDataset:
        """Get a dataset instance for a specific pipeline type."""
        if pipeline_type not in cls._datasets:
            raise KeyError(f"No dataset registered for pipeline type: {pipeline_type}")

        dataset_class = cls._datasets[pipeline_type]
        return dataset_class.from_config(config)

    @classmethod
    def get_dataset_from_alias(cls, alias: str, pipeline_type: PipelineType | None = None) -> BaseDataset:
        """Get a dataset instance from an alias."""
        # check if alias is a registered alias
        if alias not in cls._aliases:
            raise ValueError(f"Unknown dataset alias: {alias}. Available: {list(cls._aliases.keys())}")

        # Check the compatible pipeline types for the alias
        pipeline_types = cls.get_alias_supported_pipeline_types(alias)

        if len(pipeline_types) > 1 and pipeline_type is None:
            raise ValueError(
                f"Dataset alias {alias} has multiple compatible pipeline types: {pipeline_types}. Please specify a pipeline type."
            )

        if len(pipeline_types) > 1 and pipeline_type not in pipeline_types:
            raise ValueError(
                f"Dataset alias {alias} does not support pipeline type {pipeline_type}. Supported types: {pipeline_types}"
            )

        if pipeline_type is None:
            pipeline_type = next(iter(pipeline_types))

        ds_config = cls.get_alias_config(alias)

        return cls.get_dataset_for_pipeline(pipeline_type=pipeline_type, config=ds_config)

    @classmethod
    def get_expected_columns(cls, pipeline_type: PipelineType) -> list[str]:
        """Get the expected columns for a specific pipeline type."""
        if pipeline_type not in cls._datasets:
            raise KeyError(f"No dataset registered for pipeline type: {pipeline_type}")

        dataset_class = cls._datasets[pipeline_type]
        return dataset_class._expected_columns

    # New methods for alias functionality
    @classmethod
    def get_alias_config(cls, alias: str) -> DatasetConfig:
        """Get the config for a dataset alias."""
        if alias not in cls._aliases:
            raise ValueError(f"Unknown dataset alias: {alias}. Available: {list(cls._aliases.keys())}")
        return cls._aliases[alias].config

    @classmethod
    def get_alias_supported_pipeline_types(cls, alias: str) -> set[PipelineType]:
        """Get supported pipeline types for a dataset alias."""
        if alias not in cls._aliases:
            raise ValueError(f"Unknown dataset alias: {alias}")
        return cls._aliases[alias].supported_pipeline_types

    @classmethod
    def validate_alias_pipeline_compatibility(cls, alias: str, pipeline_type: PipelineType) -> None:
        """Validate that a dataset alias supports a pipeline type."""
        supported_types = cls.get_alias_supported_pipeline_types(alias)
        if pipeline_type not in supported_types:
            raise ValueError(
                f"Dataset alias '{alias}' does not support pipeline type '{pipeline_type.value}'. "
                f"Supported types: {[t.value for t in supported_types]}"
            )

    @classmethod
    def list_aliases(cls) -> list[str]:
        """List all available dataset aliases."""
        return list(cls._aliases.keys())

    @classmethod
    def list_aliases_by_pipeline_type(cls, pipeline_type: PipelineType) -> list[str]:
        """List dataset aliases that support a specific pipeline type."""
        return [alias for alias, info in cls._aliases.items() if pipeline_type in info.supported_pipeline_types]

    @classmethod
    def get_alias_info(cls, alias: str) -> DatasetAliasInfo:
        """Get the full info for a dataset alias."""
        if alias not in cls._aliases:
            raise ValueError(f"Unknown dataset alias: {alias}")
        return cls._aliases[alias]

    @classmethod
    def has_alias(cls, alias: str) -> bool:
        """Check if a dataset alias exists."""
        return alias in cls._aliases


# Register all datasets
DatasetRegistry.register(PipelineType.DIARIZATION, DiarizationDataset)
DatasetRegistry.register(PipelineType.ORCHESTRATION, OrchestrationDataset)
DatasetRegistry.register(PipelineType.STREAMING_TRANSCRIPTION, StreamingDataset)
DatasetRegistry.register(PipelineType.TRANSCRIPTION, TranscriptionDataset)
