# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Oruk Fourier 2B transcription pipelines (local Hugging Face and hosted API)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import TranscriptionSample
from ...engine.oruk_engine import (
    DEFAULT_ORUK_API_MODEL,
    DEFAULT_ORUK_HF_MODEL_ID,
    DEFAULT_ORUK_HF_REVISION,
    OrukApi,
    OrukFourierEngine,
)
from ...pipeline import Pipeline, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import TranscriptionConfig, TranscriptionOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("temp_audio_dir")


@dataclass(frozen=True)
class _OrukTranscriptionInput:
    audio_path: Path
    keywords_prompt: str | None
    language: str | None


class OrukFourierTranscriptionConfig(TranscriptionConfig):
    """Configuration for local Hugging Face ``oruk/oruk-fourier-2b`` transcription."""

    model_id: str = Field(
        default=DEFAULT_ORUK_HF_MODEL_ID,
        description="Hugging Face repo ID or local directory for Oruk Fourier 2B.",
    )
    revision: str | None = Field(
        default=DEFAULT_ORUK_HF_REVISION,
        description="Pinned Hugging Face revision for reproducible weights.",
    )
    device: str | None = Field(
        default=None,
        description="Torch device ('cuda', 'mps', 'cpu'). Auto-detects when None.",
    )
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = Field(
        default="auto",
        description="Model dtype ('auto' selects bfloat16 on CUDA/MPS and float32 on CPU).",
    )
    decode_mode: Literal["published", "greedy"] = Field(
        default="published",
        description="Decode mode: 'published' preserves generation_config.json (3-beam); 'greedy' forces num_beams=1.",
    )
    max_new_tokens: int | None = Field(
        default=None,
        description="Optional override for max_new_tokens (defaults to generation_config.json).",
    )
    chunk_length_s: float = Field(
        default=30.0,
        description="Maximum audio window in seconds per chunk (Qwen3-ASR supports up to 30s).",
    )
    cache_dir: str | None = Field(
        default=None,
        description="Optional Hugging Face cache directory.",
    )


@register_pipeline
class OrukFourierTranscriptionPipeline(Pipeline):
    """Local Hugging Face Transformers pipeline for Oruk Fourier 2B (CC BY-NC 4.0, English-only)."""

    _config_class = OrukFourierTranscriptionConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[_OrukTranscriptionInput], str]:
        engine = OrukFourierEngine(
            model_id=self.config.model_id,
            revision=self.config.revision,
            device=self.config.device,
            dtype=self.config.dtype,
            decode_mode=self.config.decode_mode,
            max_new_tokens=self.config.max_new_tokens,
            chunk_length_s=self.config.chunk_length_s,
            cache_dir=self.config.cache_dir,
        )

        def transcribe(sample_input: _OrukTranscriptionInput) -> str:
            try:
                return engine.transcribe(
                    sample_input.audio_path,
                    prompt=sample_input.keywords_prompt,
                    language=sample_input.language,
                )
            finally:
                sample_input.audio_path.unlink(missing_ok=True)

        return transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> _OrukTranscriptionInput:
        keywords_prompt = None
        if self.config.use_keywords:
            keywords = input_sample.extra_info.get("dictionary", [])
            if keywords:
                keywords_prompt = ", ".join(keywords)

        language = input_sample.language if self.config.force_language else None
        audio_path = input_sample.save_audio(TEMP_AUDIO_DIR)
        return _OrukTranscriptionInput(
            audio_path=audio_path,
            keywords_prompt=keywords_prompt,
            language=language,
        )

    def parse_output(self, output: str) -> TranscriptionOutput:
        return TranscriptionOutput(prediction=Transcript.from_words_info(words=output.split()))


class OrukApiTranscriptionConfig(TranscriptionConfig):
    """Configuration for hosted Oruk Speech API transcription (``https://speech-api.oruk.ai/v1``)."""

    model_version: str = Field(
        default=DEFAULT_ORUK_API_MODEL,
        description="Oruk Speech API model ID (default: 'oruk-fourier-2b').",
    )
    base_url: str | None = Field(
        default=None,
        description="Optional API base URL override (defaults to ORUK_BASE_URL or https://speech-api.oruk.ai/v1).",
    )
    timeout: float = Field(
        default=300.0,
        description="HTTP request timeout in seconds.",
    )


@register_pipeline
class OrukApiTranscriptionPipeline(Pipeline):
    """Hosted OpenAI-compatible API pipeline for Oruk Fourier 2B."""

    _config_class = OrukApiTranscriptionConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[_OrukTranscriptionInput], str]:
        api = OrukApi(
            model=self.config.model_version,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

        def transcribe(sample_input: _OrukTranscriptionInput) -> str:
            try:
                return api.transcribe(
                    sample_input.audio_path,
                    language=sample_input.language,
                )
            finally:
                sample_input.audio_path.unlink(missing_ok=True)

        return transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> _OrukTranscriptionInput:
        if self.config.use_keywords and input_sample.extra_info.get("dictionary"):
            logger.warning(
                "OrukApiTranscriptionPipeline does not support keyword prompts on /v1/audio/transcriptions; "
                "ignoring sample dictionary. Use `oruk-fourier-2b` (local Transformers) for keyword biasing."
            )

        language = input_sample.language if self.config.force_language else None
        audio_path = input_sample.save_audio(TEMP_AUDIO_DIR)
        return _OrukTranscriptionInput(
            audio_path=audio_path,
            keywords_prompt=None,
            language=language,
        )

    def parse_output(self, output: str) -> TranscriptionOutput:
        return TranscriptionOutput(prediction=Transcript.from_words_info(words=output.split()))
