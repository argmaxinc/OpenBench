# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from dataclasses import dataclass
from typing import Any, ClassVar

from ..types import PipelineType
from .base import Pipeline, PipelineConfig


@dataclass
class PipelineAliasInfo:
    """Information about a pipeline alias."""

    pipeline_class: type[Pipeline]
    default_config: dict[str, Any]
    description: str = ""


class PipelineRegistry:
    """Registry for pipelines with support for aliases and default configurations."""

    _pipelines: ClassVar[dict[str, type[Pipeline]]] = {}
    _aliases: ClassVar[dict[str, PipelineAliasInfo]] = {}

    @classmethod
    def register(cls, pipeline_class: type[Pipeline]) -> None:
        """Register a pipeline class by its class name."""
        cls._pipelines[pipeline_class.__name__] = pipeline_class

    @classmethod
    def register_alias(
        cls,
        alias: str,
        pipeline_class: type[Pipeline],
        default_config: dict[str, Any],
        description: str = "",
    ) -> None:
        """Register a pipeline alias with optional default configuration."""
        # Prevent aliases from being the same as class names to avoid ambiguity
        if alias == pipeline_class.__name__:
            raise ValueError(
                f"Alias '{alias}' cannot be the same as the pipeline class name '{pipeline_class.__name__}'. "
                f"Please choose a different alias name."
            )

        # Check if alias already exists
        if alias in cls._aliases:
            raise ValueError(f"Alias '{alias}' is already registered.")

        cls._aliases[alias] = PipelineAliasInfo(
            pipeline_class=pipeline_class,
            default_config=default_config,
            description=description,
        )

    @classmethod
    def get_pipeline_class(cls, name: str) -> type[Pipeline]:
        """Get a pipeline class by name (class name or alias)."""
        # First check if it's a registered alias
        if name in cls._aliases:
            return cls._aliases[name].pipeline_class

        # Then check if it's a registered pipeline class name
        if name in cls._pipelines:
            return cls._pipelines[name]

        raise ValueError(
            f"Unknown pipeline: {name}. Available: {list(cls._pipelines.keys()) + list(cls._aliases.keys())}"
        )

    @classmethod
    def create_pipeline(cls, name: str, config: dict[str, Any] | None = None) -> Pipeline:
        """Create a pipeline instance by name (class name or alias)."""
        pipeline_class = cls.get_pipeline_class(name)

        # Handle config merging for aliases
        if name in cls._aliases:
            # Start with default config
            final_config = cls._aliases[name].default_config.copy()
            # Override with provided config if any
            if config is not None:
                final_config.update(config)
        elif config is None:
            raise ValueError(
                f"No configuration provided for pipeline '{name}' and it's not a registered alias. "
                f"Please provide a configuration or use a registered alias with default config."
            )
        else:
            final_config = config

        # Create config instance using the pipeline's config class
        return pipeline_class.from_dict(final_config)

    @classmethod
    def get_pipeline_class_for_config(cls, config_type: type[PipelineConfig]) -> type[Pipeline]:
        """Return the registered Pipeline subclass whose `_config_class` is `config_type`.

        Lets a caller inspect a pipeline's type (e.g. `pipeline_type`) before
        paying the cost of instantiation.
        """
        for pipeline_class in cls._pipelines.values():
            if pipeline_class._config_class is config_type:
                return pipeline_class
        raise ValueError(
            f"No registered pipeline has _config_class == {config_type.__name__}. "
            f"Registered configs: {[c._config_class.__name__ for c in cls._pipelines.values()]}"
        )

    @classmethod
    def create_pipeline_from_config(cls, config: PipelineConfig) -> Pipeline:
        """Instantiate the pipeline whose `_config_class` matches `type(config)`.

        Useful when a caller already has a typed config in hand (e.g. a metric
        that needs to spin up a transcription pipeline) and doesn't want to
        construct the Pipeline subclass directly.
        """
        return cls.get_pipeline_class_for_config(type(config))(config)

    @classmethod
    def get_pipeline_type(cls, name: str) -> PipelineType:
        """Get the pipeline type for a given pipeline name."""
        pipeline_class = cls.get_pipeline_class(name)
        return pipeline_class.pipeline_type

    @classmethod
    def list_pipelines(cls) -> list[str]:
        """List all available pipeline names (class names and aliases)."""
        return list(cls._pipelines.keys()) + list(cls._aliases.keys())

    @classmethod
    def list_pipelines_by_type(cls, pipeline_type: PipelineType) -> list[str]:
        """List pipeline names that match a specific pipeline type."""
        pipelines = []
        for name in cls.list_pipelines():
            if cls.get_pipeline_type(name) == pipeline_type:
                pipelines.append(name)
        return pipelines

    @classmethod
    def get_alias_info(cls, alias: str) -> PipelineAliasInfo:
        """Get the full info for a pipeline alias."""
        if alias not in cls._aliases:
            raise ValueError(f"Unknown pipeline alias: {alias}")
        return cls._aliases[alias]

    @classmethod
    def is_alias(cls, name: str) -> bool:
        """Check if a name is a registered alias."""
        return name in cls._aliases

    @classmethod
    def get_default_config(cls, name: str) -> dict[str, Any]:
        """Get the default configuration for a pipeline (if it's an alias)."""
        if name not in cls._aliases:
            raise ValueError(f"No default config available for pipeline: {name}")
        return cls._aliases[name].default_config.copy()
