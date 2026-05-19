# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import Callable, ClassVar, Type, TypeVar

from pyannote.metrics.base import BaseMetric
from pyannote.metrics.detection import (
    DetectionCostFunction,
    DetectionErrorRate,
    DetectionPrecisionRecallFMeasure,
)
from pyannote.metrics.diarization import (
    DiarizationErrorRate,
    DiarizationHomogeneity,
    DiarizationPurity,
    DiarizationPurityCoverageFMeasure,
    JaccardErrorRate,
)

from ..types import PipelineType
from .metric import MetricOptions


T = TypeVar("T", bound=BaseMetric)


class MetricRegistry:
    """Registry for metrics by pipeline type.

    This registry allows registering and retrieving metrics based on pipeline type.
    The same MetricOption (e.g. WER) can map to different metric classes for
    different pipeline types — for example a SPEECH_GENERATION pipeline can register
    a WER implementation that first transcribes generated audio, while
    TRANSCRIPTION uses the standard transcript-vs-transcript implementation.
    """

    # Pipeline type -> Metric option -> Metric class
    _metrics: ClassVar[dict[PipelineType, dict[MetricOptions, Type[BaseMetric]]]] = {}

    @classmethod
    def register(
        cls,
        metric_class: Type[T],
        pipeline_types: PipelineType | tuple[PipelineType, ...],
        metric_option: MetricOptions,
    ) -> None:
        """Register a metric class for specific pipeline types and metric option.

        Args:
            metric_class: The metric class to register
            pipeline_types: Single pipeline type or tuple of pipeline types this metric supports
            metric_option: The metric option to register
        """
        # Convert single pipeline type to tuple
        if isinstance(pipeline_types, PipelineType):
            pipeline_types = (pipeline_types,)

        for pipeline_type in pipeline_types:
            cls._metrics.setdefault(pipeline_type, {})[metric_option] = metric_class

    @classmethod
    def register_metric(
        cls,
        pipeline_types: PipelineType | tuple[PipelineType, ...],
        metric_option: MetricOptions,
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a metric class.

        This method can be used as a decorator to register custom metric classes.
        It provides a more convenient way to register metrics compared to calling
        register() directly.

        Example:
            # Single pipeline type
            @MetricRegistry.register_metric(PipelineType.DIARIZATION, MetricOptions.CUSTOM_METRIC)
            class CustomMetric(BaseMetric):
                pass

            # Multiple pipeline types
            @MetricRegistry.register_metric(
                (PipelineType.DIARIZATION, PipelineType.ORCHESTRATION),
                MetricOptions.CUSTOM_METRIC
            )
            class AnotherCustomMetric(BaseMetric):
                pass

        Args:
            pipeline_types: Single pipeline type or tuple of pipeline types this metric supports
            metric_option: The metric option to register

        Returns:
            A decorator function that registers the metric class
        """

        def decorator(metric_class: Type[T]) -> Type[T]:
            cls.register(metric_class, pipeline_types, metric_option)
            return metric_class

        return decorator

    @classmethod
    def get_metric(cls, pipeline_type: PipelineType, metric_option: MetricOptions, **kwargs) -> BaseMetric:
        """Get a metric instance for a specific pipeline type and metric option.

        Args:
            pipeline_type: The pipeline type the metric is being requested for
            metric_option: The metric option to get
            **kwargs: Additional arguments to pass to the metric constructor

        Returns:
            An instance of the requested metric

        Raises:
            KeyError: If the metric is not registered for the given pipeline type
        """
        pipeline_metrics = cls._metrics.get(pipeline_type)
        if pipeline_metrics is None or metric_option not in pipeline_metrics:
            raise KeyError(f"Metric {metric_option} not registered for pipeline type {pipeline_type}")

        return pipeline_metrics[metric_option](**kwargs)

    @classmethod
    def get_available_metrics(cls, pipeline_type: PipelineType) -> list[MetricOptions]:
        """Get all available metrics for a specific pipeline type.

        Args:
            pipeline_type: The type of pipeline

        Returns:
            List of available metric options
        """
        return list(cls._metrics.get(pipeline_type, {}).keys())


# Register all existing and interesting metrics from pyannote.metrics
# Custom metrics will be registered in their own files
MetricRegistry.register(DiarizationErrorRate, PipelineType.DIARIZATION, MetricOptions.DER)
MetricRegistry.register(JaccardErrorRate, PipelineType.DIARIZATION, MetricOptions.JER)
MetricRegistry.register(DiarizationPurity, PipelineType.DIARIZATION, MetricOptions.DIARIZATION_PURITY)
MetricRegistry.register(
    DiarizationPurityCoverageFMeasure,
    PipelineType.DIARIZATION,
    MetricOptions.DIARIZATION_PURITY_COVERAGE_FMEASURE,
)
MetricRegistry.register(
    DiarizationHomogeneity,
    PipelineType.DIARIZATION,
    MetricOptions.DIARIZATION_HOMOGENEITY,
)
MetricRegistry.register(DetectionErrorRate, PipelineType.DIARIZATION, MetricOptions.DETECTION_ERROR_RATE)
MetricRegistry.register(
    DetectionCostFunction,
    PipelineType.DIARIZATION,
    MetricOptions.DETECTION_COST_FUNCTION,
)
MetricRegistry.register(
    DetectionPrecisionRecallFMeasure,
    PipelineType.DIARIZATION,
    MetricOptions.DETECTION_PRECISION_RECALL_FMEASURE,
)
