# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Speech-generation pipeline using the WhisperKit CLI.

The pipeline shells out to `whisperkit-cli tts` to synthesize a WAV from
a text prompt and returns a `GeneratedAudio` prediction containing the
file path and the measured duration. Transcription / WER scoring is
performed by the `SpeechGenerationWordErrorRate` metric, not here, so
the pipeline's reported `prediction_time` reflects TTS only.
"""

import subprocess
from pathlib import Path
from typing import Callable

import librosa
from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset.dataset_speech_generation import SpeechGenerationSample
from ...pipeline_prediction import GeneratedAudio
from ..base import (
    Pipeline,
    PipelineConfig,
    PipelineOutput,
    PipelineType,
    register_pipeline,
)


logger = get_logger(__name__)

TEMP_TTS_AUDIO_DIR = Path("./temp_tts_audio")


class WhisperKitSpeechGenerationConfig(PipelineConfig):
    """Config for the WhisperKit speech-generation pipeline.

    All fields are specific to whisperkit-cli; we don't currently share
    anything across TTS providers, so this config lives next to the
    pipeline rather than under a common base.
    """

    cli_path: str = Field(
        ...,
        description="Path to the whisperkit-cli binary (used for TTS generation).",
    )
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


class SpeechGenerationInput(BaseModel):
    """Input for the speech generation pipeline."""

    text: str = Field(
        ...,
        description="Text prompt to generate speech from.",
    )
    audio_name: str = Field(
        ...,
        description="Unique identifier for this sample (used for temp file naming).",
    )


@register_pipeline
class WhisperKitSpeechGenerationPipeline(Pipeline):
    """Speech-generation pipeline using the WhisperKit CLI.

    For each sample, the pipeline:

    1. Builds a `whisperkit-cli tts` invocation from the config.
    2. Synthesizes the WAV to a temp directory under cwd.
    3. Measures the duration via librosa.
    4. Returns a `GeneratedAudio` prediction (path + duration).

    The generated WAV is left in place so the WER metric can transcribe
    it and the wandb logger can copy it into the predictions artifact.
    On TTS failure the partial WAV (if any) is removed so the temp dir
    doesn't accumulate stale files across retries.
    """

    _config_class = WhisperKitSpeechGenerationConfig
    pipeline_type = PipelineType.SPEECH_GENERATION

    def build_pipeline(self) -> Callable[[SpeechGenerationInput], GeneratedAudio]:
        config = self.config

        def generate(input: SpeechGenerationInput) -> GeneratedAudio:
            TEMP_TTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            audio_path = TEMP_TTS_AUDIO_DIR / f"{input.audio_name}.wav"

            tts_cmd = [
                config.cli_path,
                "tts",
                "--text",
                input.text,
                "--speaker",
                config.speaker,
                "--language",
                config.language,
                "--output-path",
                str(audio_path),
                "--temperature",
                str(config.temperature),
                "--top-k",
                str(config.top_k),
                "--max-new-tokens",
                str(config.max_new_tokens),
            ]
            if config.seed is not None:
                tts_cmd.extend(["--seed", str(config.seed)])
            if config.models_path is not None:
                tts_cmd.extend(["--models-path", config.models_path])
            if config.model_repo is not None:
                tts_cmd.extend(["--model-repo", config.model_repo])
            if config.version_dir is not None:
                tts_cmd.extend(["--version-dir", config.version_dir])
            if config.tokenizer is not None:
                tts_cmd.extend(["--tokenizer", config.tokenizer])

            logger.debug(f"Running TTS: {' '.join(tts_cmd)}")

            try:
                tts_result = subprocess.run(tts_cmd, capture_output=True, text=True)
                if tts_result.returncode != 0:
                    raise RuntimeError(
                        "whisperkit-cli tts failed "
                        f"(exit {tts_result.returncode}):\n"
                        f"  stdout: {tts_result.stdout[:500]}\n"
                        f"  stderr: {tts_result.stderr[:500]}"
                    )
                if not audio_path.exists():
                    raise RuntimeError(f"TTS completed but audio file not found at {audio_path}")

                duration = float(librosa.get_duration(path=str(audio_path)))
                logger.info(f"Generated TTS audio: {audio_path} ({duration:.2f}s)")

                return GeneratedAudio(
                    audio_path=str(audio_path),
                    duration=duration,
                )
            except Exception:
                # On any failure during synthesis or duration probing,
                # clean up the partial WAV so the temp dir doesn't grow.
                audio_path.unlink(missing_ok=True)
                raise

        return generate

    def parse_input(self, input_sample: SpeechGenerationSample) -> SpeechGenerationInput:
        """Extract the text prompt and a stable identifier from the sample."""
        text = input_sample.reference.get_transcript_string()
        return SpeechGenerationInput(
            text=text,
            audio_name=input_sample.audio_name,
        )

    def parse_output(self, output: GeneratedAudio) -> PipelineOutput[GeneratedAudio]:
        """Wrap the generated audio in a `PipelineOutput`."""
        return PipelineOutput[GeneratedAudio](prediction=output)
