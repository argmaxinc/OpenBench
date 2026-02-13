# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from pydantic import Field

from ...pipeline_prediction import Transcript
from ..base import PipelineConfig, PipelineOutput


class SpeechGenerationConfig(PipelineConfig):
    """Base config for speech generation pipelines."""

    cli_path: str = Field(
        ...,
        description=("Path to the whisperkit-cli binary (used for TTS generation)."),
    )

    # TTS parameters
    speaker: str = Field(
        default="aiden",
        description="Speaker voice for TTS generation.",
    )
    language: str = Field(
        default="english",
        description="Language for TTS generation.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible output.",
    )
    temperature: float = Field(
        default=0.9,
        description="Sampling temperature for TTS.",
    )
    top_k: int = Field(
        default=50,
        description="Top-k sampling for TTS.",
    )
    max_new_tokens: int = Field(
        default=245,
        description="Max RVQ frames to generate.",
    )
    models_path: str | None = Field(
        default=None,
        description="Local model directory for TTS.",
    )
    model_repo: str | None = Field(
        default=None,
        description="HF repo for TTS model download.",
    )
    version_dir: str | None = Field(
        default=None,
        description="TTS model version directory.",
    )
    tokenizer: str | None = Field(
        default=None,
        description="HF tokenizer repo or local path.",
    )

    # Transcription parameters
    transcription_cli_path: str | None = Field(
        default=None,
        description=("Path to CLI for transcription. Defaults to cli_path if not set."),
    )
    transcription_repo_id: str | None = Field(
        default=None,
        description=("HuggingFace repo ID for transcription model (e.g. argmaxinc/parakeetkit-pro)."),
    )
    transcription_model_variant: str | None = Field(
        default=None,
        description=("Model variant folder within the repo (e.g. nvidia_parakeet-v2_476MB)."),
    )
    transcription_model_path: str | None = Field(
        default=None,
        description=("Local path to ASR model dir. Overrides repo_id/model_variant."),
    )
    transcription_word_timestamps: bool = Field(
        default=True,
        description="Include word timestamps.",
    )
    transcription_chunking_strategy: str = Field(
        default="vad",
        description="Chunking strategy (none or vad).",
    )


class SpeechGenerationOutput(PipelineOutput[Transcript]):
    """Output for speech generation pipelines.

    The prediction is a Transcript of the generated audio
    (obtained by transcribing the TTS output). WER is
    computed against the original text prompt.
    """

    pass
