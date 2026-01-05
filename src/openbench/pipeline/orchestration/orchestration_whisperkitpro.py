# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from pathlib import Path
from typing import Literal

from argmaxtools.utils import get_logger
from coremltools import ComputeUnit
from pydantic import Field

from ...dataset import OrchestrationSample
from ...engine import WhisperKitPro, WhisperKitProConfig, WhisperKitProInput, WhisperKitProOutput
from ...pipeline_prediction import Transcript, Word
from ..base import Pipeline, PipelineType, register_pipeline
from .common import OrchestrationConfig, OrchestrationOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("./temp_audio")


class WhisperKitProOrchestrationConfig(OrchestrationConfig):
    """Configuration for WhisperKitPro orchestration pipeline.

    Supports two modes:
    1. Legacy: model_version, model_prefix, model_repo_name
    2. New: repo_id, model_variant (downloads and manages models)
    """

    cli_path: str = Field(
        ...,
        description="The path to the WhisperKitPro CLI",
    )

    # Legacy fields (optional)
    model_version: str | None = Field(
        None,
        description="(Legacy) The version of the WhisperKitPro model to use",
    )
    model_prefix: str | None = Field(
        None,
        description="(Legacy) The prefix of the model to use.",
    )
    model_repo_name: str | None = Field(
        None,
        description="(Legacy) The name of the Hugging Face model repo to use.",
    )

    # New fields for model download and management
    repo_id: str | None = Field(
        None,
        description="HuggingFace repo ID",
    )
    model_variant: str | None = Field(
        None,
        description="Model variant folder name",
    )
    model_dir: str | None = Field(
        None,
        description="Local path to model directory. If provided, models are loaded from this path directly instead of downloading.",
    )

    audio_encoder_compute_units: ComputeUnit = Field(
        ComputeUnit.CPU_AND_NE,
        description="The compute units to use for the audio encoder. Default is CPU_AND_NE.",
    )
    text_decoder_compute_units: ComputeUnit = Field(
        ComputeUnit.CPU_AND_NE,
        description="The compute units to use for the text decoder. Default is CPU_AND_NE.",
    )
    orchestration_strategy: Literal["word", "segment"] = Field(
        "segment",
        description="The orchestration strategy to use either `word` or `segment`",
    )
    clusterer_version: Literal["pyannote3", "pyannote4"] = Field(
        "pyannote4",
        description="The version of the clusterer to use",
    )
    use_exclusive_reconciliation: bool = Field(
        False,
        description="Whether to use exclusive reconciliation",
    )
    fast_load: bool = Field(
        False,
        description="Whether to use fast load",
    )


@register_pipeline
class WhisperKitProOrchestrationPipeline(Pipeline):
    _config_class = WhisperKitProOrchestrationConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self) -> WhisperKitPro:
        whisperkitpro_config = WhisperKitProConfig(
            model_version=self.config.model_version,
            model_prefix=self.config.model_prefix,
            model_repo_name=self.config.model_repo_name,
            repo_id=self.config.repo_id,
            model_variant=self.config.model_variant,
            model_dir=self.config.model_dir,
            audio_encoder_compute_units=self.config.audio_encoder_compute_units,
            text_decoder_compute_units=self.config.text_decoder_compute_units,
            report_path="whisperkitpro_orchestration_reports",
            word_timestamps=True,
            chunking_strategy="vad",
            diarization=True,
            orchestration_strategy=self.config.orchestration_strategy,
            clusterer_version_string=self.config.clusterer_version,
            use_exclusive_reconciliation=self.config.use_exclusive_reconciliation,
            fast_load=self.config.fast_load,
        )
        # Create WhisperKit engine
        engine = WhisperKitPro(
            cli_path=self.config.cli_path,
            transcription_config=whisperkitpro_config,
        )

        return engine

    def parse_input(self, input_sample: OrchestrationSample) -> WhisperKitProInput:
        # Extract language if force_language is enabled
        language = None
        if self.config.force_language:
            language = input_sample.language

        return WhisperKitProInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            keep_audio=False,
            language=language,
        )

    def parse_output(self, output: WhisperKitProOutput) -> OrchestrationOutput:
        rttm_path = output.rttm_report_path
        # Create words
        words: list[Word] = []
        for line in rttm_path.read_text().splitlines():
            parts = line.split()
            speaker = parts[-3]
            transcript_words = parts[5:-4]
            words.extend(
                [
                    Word(
                        word=word,
                        speaker=speaker,
                        start=None,
                        end=None,
                    )
                    for word in transcript_words
                ]
            )

        prediction = Transcript(words=words)
        return OrchestrationOutput(prediction=prediction)
