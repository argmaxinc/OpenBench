# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
from pathlib import Path

from argmaxtools.utils import get_logger
from coremltools import ComputeUnit
from pydantic import Field

from ...dataset import TranscriptionSample
from ...engine import WhisperKitPro, WhisperKitProConfig, WhisperKitProInput, WhisperKitProOutput
from ...pipeline_prediction import Transcript
from ..base import Pipeline, PipelineType, register_pipeline
from .common import TranscriptionConfig, TranscriptionOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("./temp_audio")
TEMP_VOCAB_DIR = Path("./temp_vocab")


class WhisperKitProTranscriptionConfig(TranscriptionConfig):
    """Configuration for WhisperKitPro transcription pipeline.

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
        description="(Legacy) WhisperKitPro model version",
    )
    model_prefix: str | None = Field(
        None,
        description="(Legacy) Model prefix",
    )
    model_repo_name: str | None = Field(
        None,
        description="(Legacy) HuggingFace model repo name",
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
    models_cache_dir: str | None = Field(
        None,
        description="Directory to cache downloaded models",
    )

    audio_encoder_compute_units: ComputeUnit = Field(
        ComputeUnit.CPU_AND_NE,
        description="The compute units to use for the audio encoder. Default is CPU_AND_NE.",
    )
    text_decoder_compute_units: ComputeUnit = Field(
        ComputeUnit.CPU_AND_NE,
        description="The compute units to use for the text decoder. Default is CPU_AND_NE.",
    )
    fast_load: bool = Field(
        False,
        description="Whether to use fast load",
    )
    custom_vocabulary_path: str | None = Field(
        None,
        description="Path to custom vocabulary file for keyword boosting",
    )


@register_pipeline
class WhisperKitProTranscriptionPipeline(Pipeline):
    _config_class = WhisperKitProTranscriptionConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> WhisperKitPro:
        whisperkitpro_config = WhisperKitProConfig(
            model_version=self.config.model_version,
            model_prefix=self.config.model_prefix,
            model_repo_name=self.config.model_repo_name,
            repo_id=self.config.repo_id,
            model_variant=self.config.model_variant,
            models_cache_dir=self.config.models_cache_dir,
            audio_encoder_compute_units=(
                self.config.audio_encoder_compute_units
            ),
            text_decoder_compute_units=(
                self.config.text_decoder_compute_units
            ),
            report_path="whisperkitpro_transcription_reports",
            word_timestamps=True,
            chunking_strategy="vad",
            diarization=False,
            fast_load=self.config.fast_load,
        )
        # Create WhisperKit engine
        engine = WhisperKitPro(
            cli_path=self.config.cli_path,
            transcription_config=whisperkitpro_config,
        )

        return engine

    def parse_input(self, input_sample: TranscriptionSample) -> WhisperKitProInput:
        """Override to extract keywords from sample before processing."""
        # Extract keywords from sample's extra_info if flag is enabled
        custom_vocab_path = None
        if self.config.use_keywords:
            keywords = input_sample.extra_info.get("dictionary", [])
            if keywords:
                # Create temp vocab directory if it doesn't exist
                TEMP_VOCAB_DIR.mkdir(parents=True, exist_ok=True)

                # Create a vocab file for this sample
                vocab_file = TEMP_VOCAB_DIR / "vocab.txt"

                # Write keywords to file (one per line)
                with vocab_file.open("w") as f:
                    f.write("\n".join(keywords))

                custom_vocab_path = str(vocab_file)
                logger.debug(f"Created custom vocabulary file: {custom_vocab_path} with {len(keywords)} keywords")

        return WhisperKitProInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            keep_audio=False,
            custom_vocabulary_path=custom_vocab_path,
        )

    def parse_output(self, output: WhisperKitProOutput) -> TranscriptionOutput:
        """Parse JSON output file into TranscriptionOutput."""
        with output.json_report_path.open("r") as f:
            data = json.load(f)

        transcript = Transcript.from_words_info(
            words=[word["word"] for segment in data["segments"] for word in segment["words"]],
            start=[word["start"] for segment in data["segments"] for word in segment["words"] if "start" in word],
            end=[word["end"] for segment in data["segments"] for word in segment["words"] if "end" in word],
        )

        return TranscriptionOutput(
            prediction=transcript,
        )
