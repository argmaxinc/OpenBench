# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from typing import Generic, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field

from ..pipeline_prediction import DiarizationAnnotation, StreamingTranscript, Transcript
from ..types import PredictionProtocol


Prediction = TypeVar("Prediction", bound=PredictionProtocol)


class BaseSampleResult(BaseModel, Generic[Prediction]):
    """The base result for a given sample of a dataset shared across all pipelines types"""

    dataset_name: str = Field(..., description="The name of the dataset")
    sample_id: int = Field(..., description="The id of the sample")
    audio_name: str = Field(..., description="The name of the audio file")
    pipeline_name: str = Field(..., description="The name of the pipeline")
    prediction: Prediction = Field(..., description="The predicted diarization result")
    prediction_time: float = Field(
        ...,
        description="The elapsed time in seconds for the pipeline to run on the sample",
    )
    audio_duration: float = Field(
        ...,
        description="The duration of the audio in seconds",
    )

    class Config:
        arbitrary_types_allowed = True


class DiarizationSampleResult(BaseSampleResult[DiarizationAnnotation]):
    """The diarization result for a given sample of a dataset"""

    embeddings: np.ndarray | None = Field(
        None,
        description="The embeddings of the diarization result if the pipeline supports it",
    )
    cluster_labels: np.ndarray | None = Field(
        None,
        description="The cluster labels of the diarization result if the pipeline supports it",
    )
    centroids: np.ndarray | None = Field(
        None,
        description="The centroids of the diarization result if the pipeline supports it",
    )
    num_speakers_predicted: int = Field(
        ...,
        description="The number of speakers predicted by the pipeline",
    )
    num_speakers_reference: int = Field(
        ...,
        description="The number of speakers in the reference annotation",
    )


class TranscriptionSampleResult(BaseSampleResult[Union[Transcript, StreamingTranscript]]):
    """The transcription result for a given sample of a dataset.
    This could be associated with the following pipeline types:
    - Transcription
    - Streaming Transcription
    - Orchestration
    """


class TaskResult(BaseModel):
    """The evaluation result for a given task on a given sample of a dataset"""

    dataset_name: str = Field(..., description="The name of the dataset")
    sample_id: int = Field(..., description="The id of the sample")
    pipeline_name: str = Field(..., description="The name of the pipeline")
    metric_name: str = Field(..., description="The name of the metric")
    result: float | None = Field(..., description="The result of the metric")
    detailed_result: dict[str, float | None] = Field(
        None,
        description="The detailed results of the metric i.e. breakdown by its components allowing \
        for more granular analysis",
    )


class GlobalResult(BaseModel):
    """The evaluation result for a given metric over all samples of a dataset"""

    dataset_name: str = Field(..., description="The name of the dataset")
    pipeline_name: str = Field(..., description="The name of the pipeline")
    metric_name: str = Field(..., description="The name of the metric")
    global_result: float | None = Field(..., description="The global result of the metric")
    detailed_result: dict[str, float | None] = Field(
        None,
        description="The detailed results of the metric i.e. breakdown by its components \
        allowing for more granular analysis",
    )
    avg_result: float | None = Field(..., description="The average result of the metric")
    upper_bound: float | None = Field(..., description="The upper bound of the confidence interval")
    lower_bound: float | None = Field(..., description="The lower bound of the confidence interval")


class BenchmarkResult(BaseModel):
    sample_results: list[DiarizationSampleResult | TranscriptionSampleResult] = Field(
        ..., description="The results of the samples"
    )
    task_results: list[TaskResult] = Field(..., description="The results of the tasks")
    global_results: list[GlobalResult] = Field(..., description="The results of the global metrics")
