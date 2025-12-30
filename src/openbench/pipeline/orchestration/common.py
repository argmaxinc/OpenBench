# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from abc import abstractmethod
from typing import Any, Callable

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import DiarizationSample
from ...pipeline_prediction import Transcript
from ..base import (
    PIPELINE_REGISTRY,
    Pipeline,
    PipelineConfig,
    PipelineOutput,
    PipelineType,
)
from ..diarization import DiarizationOutput
from ..transcription import TranscriptionOutput


logger = get_logger(__name__)


class OrchestrationConfig(PipelineConfig):
    """Base configuration for orchestration pipelines."""

    force_language: bool = Field(
        False,
        description="Force the language of the audio files i.e. hint the model to use the correct language.",
    )


class OrchestrationOutput(PipelineOutput[Transcript]):
    transcription_output: TranscriptionOutput | None = Field(
        default=None,
        description="The transcription output from the ASR pipeline",
    )
    diarization_output: DiarizationOutput | None = Field(
        default=None,
        description="The diarization output from the diarization pipeline",
    )


def create_pipeline(name: str, config_dict: dict[str, Any]) -> Pipeline:
    pipeline_cls = PIPELINE_REGISTRY[name]
    pipeline_config_cls = pipeline_cls._config_class
    pipeline_config = pipeline_config_cls(**config_dict)
    return pipeline_cls(pipeline_config)


# Orchestration pipeline has three distinct cases:
# 1. Where we run indepedently the ASR and the diarization pipeline and then merge the results
# 2. Where we leverage VAD or Diarization Pipeline to generate speech segments and then run ASR on each segment
# 3. Black box systems where it will work as a wrapper for an existing provider e.g. WhisperX
# they can have their own implemenation files


# Case - 1
class PostInferenceMergePipelineConfig(PipelineConfig):
    diarization_pipeline_name: str
    diarization_pipeline_config: dict[str, Any]
    asr_pipeline_name: str
    asr_pipeline_config: dict[str, Any]


class PostInferenceMergePipeline(Pipeline):
    _config_class = PostInferenceMergePipelineConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(
        self,
    ) -> Callable[[DiarizationSample], tuple[DiarizationOutput, TranscriptionOutput]]:
        diarization_pipeline = create_pipeline(
            name=self.config.diarization_pipeline_name,
            config_dict=self.config.diarization_pipeline_config,
        )
        asr_pipeline = create_pipeline(
            name=self.config.asr_pipeline_name,
            config_dict=self.config.asr_pipeline_config,
        )

        def indepdent_inference(
            input_sample: DiarizationSample,
        ) -> tuple[DiarizationOutput, TranscriptionOutput]:
            diarization_output = diarization_pipeline(input_sample)
            transcription_output = asr_pipeline(input_sample)
            return diarization_output, transcription_output

        return indepdent_inference

    def parse_input(self, input_sample: DiarizationSample) -> DiarizationOutput:
        return input_sample.diarization_output

    # This is where the merging strategies should be implemented
    @abstractmethod
    def parse_output(self, output: tuple[DiarizationOutput, TranscriptionOutput]) -> OrchestrationOutput: ...
