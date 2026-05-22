# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Speech-generation pipeline via Argmax SDK open-source `argmax-cli tts`."""

from pathlib import Path
from typing import Callable, Literal

import librosa
from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset.dataset_speech_generation import SpeechGenerationSample
from ...engine.argmax_oss_engine import (
    ArgmaxOpenSourceEngine,
    ArgmaxOpenSourceEngineConfig,
    TtsCliInput,
    TtsCliOutput,
)
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


class ArgmaxOpenSourceSpeechGenerationConfig(PipelineConfig):
    """Config for the Argmax OSS speech-generation pipeline.

    Engine fields (cache_dir / commit_hash / cli_path) mirror the
    transcription / diarization argmax-oss configs. TTS-specific fields
    follow `argmax-cli tts` flags.
    """

    cache_dir: str | None = Field(
        default=None,
        description="Cache directory for argmax-oss clone + CLI build. "
        "Defaults to ARGMAX_OSS_CACHE_DIR or ~/.cache/openbench/argmax-oss.",
    )
    commit_hash: str | None = Field(
        default=None,
        description="Optional git commit pin for the clone.",
    )
    cli_path: str | None = Field(
        default=None,
        description="Prebuilt argmax-cli path; skips clone/build.",
    )
    speaker: Literal[
        "ryan",
        "aiden",
        "ono-anna",
        "sohee",
        "eric",
        "dylan",
        "serena",
        "vivian",
        "uncle-fu",
    ] = Field(default="aiden", description="--speaker.")
    language: Literal[
        "english",
        "chinese",
        "japanese",
        "korean",
        "german",
        "french",
        "russian",
        "portuguese",
        "spanish",
        "italian",
    ] = Field(default="english", description="--language.")
    output_format: Literal["wav", "m4a"] = Field(
        default="wav",
        description="--output-format. WAV is preferred so the WER metric can decode without extra deps.",
    )
    seed: int | None = Field(default=None, description="--seed for reproducible output.")
    temperature: float = Field(default=0.9, description="--temperature.")
    top_k: int = Field(default=50, description="--top-k.")
    max_new_tokens: int = Field(default=245, description="--max-new-tokens (RVQ frames).")
    instruction: str | None = Field(
        default=None,
        description="--instruction (style hint, e.g. 'Speak slowly'). 1.7B model only.",
    )
    model: Literal["0.6b", "1.7b"] | None = Field(
        default=None,
        description="--model preset (0.6b or 1.7b). Leave None to use CLI default.",
    )
    models_path: str | None = Field(default=None, description="--models-path (local model dir).")
    model_repo: str | None = Field(default=None, description="--model-repo (HF repo).")
    version_dir: str | None = Field(default=None, description="--version-dir (overrides --model preset).")
    tokenizer: str | None = Field(default=None, description="--tokenizer (HF repo or local path).")

    def generate_tts_cli_args(self) -> list[str]:
        args: list[str] = [
            "--speaker",
            self.speaker,
            "--language",
            self.language,
            "--output-format",
            self.output_format,
            "--temperature",
            str(self.temperature),
            "--top-k",
            str(self.top_k),
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]
        if self.seed is not None:
            args.extend(["--seed", str(self.seed)])
        if self.instruction is not None:
            args.extend(["--instruction", self.instruction])
        if self.model is not None:
            args.extend(["--model", self.model])
        if self.models_path is not None:
            args.extend(["--models-path", self.models_path])
        if self.model_repo is not None:
            args.extend(["--model-repo", self.model_repo])
        if self.version_dir is not None:
            args.extend(["--version-dir", self.version_dir])
        if self.tokenizer is not None:
            args.extend(["--tokenizer", self.tokenizer])
        return args


class SpeechGenerationInput(BaseModel):
    """Input for the speech-generation pipeline."""

    text: str = Field(..., description="Text prompt to generate speech from.")
    audio_name: str = Field(..., description="Unique identifier for this sample (used for temp file naming).")


@register_pipeline
class ArgmaxOpenSourceSpeechGenerationPipeline(Pipeline):
    """Speech-generation pipeline using `argmax-cli tts` via `ArgmaxOpenSourceEngine`.

    For each sample, the pipeline:

    1. Builds an `argmax-cli tts` flag list from the config.
    2. Synthesizes audio to a temp directory under cwd.
    3. Measures the duration via librosa.
    4. Returns a `GeneratedAudio` prediction (path + duration).

    WER scoring is performed by `SpeechGenerationWordErrorRate`, not here,
    so the pipeline's reported `prediction_time` reflects TTS only.
    """

    _config_class = ArgmaxOpenSourceSpeechGenerationConfig
    pipeline_type = PipelineType.SPEECH_GENERATION

    def build_pipeline(self) -> Callable[[SpeechGenerationInput], GeneratedAudio]:
        engine = ArgmaxOpenSourceEngine(
            ArgmaxOpenSourceEngineConfig(
                cache_dir=self.config.cache_dir,
                commit_hash=self.config.commit_hash,
                cli_path=self.config.cli_path,
            )
        )
        tts_args = self.config.generate_tts_cli_args()
        suffix = f".{self.config.output_format}"

        def generate(inp: SpeechGenerationInput) -> GeneratedAudio:
            TEMP_TTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            audio_path = TEMP_TTS_AUDIO_DIR / f"{inp.audio_name}{suffix}"
            try:
                output: TtsCliOutput = engine.tts(
                    TtsCliInput(text=inp.text, output_path=audio_path),
                    tts_args,
                )
                duration = float(librosa.get_duration(path=str(output.audio_path)))
                logger.debug("Generated TTS audio: %s (%.2fs)", output.audio_path, duration)
                return GeneratedAudio(
                    audio_path=str(output.audio_path),
                    duration=duration,
                )
            except Exception:
                # Clean up partial output so the temp dir doesn't grow across retries.
                audio_path.unlink(missing_ok=True)
                raise

        return generate

    def parse_input(self, input_sample: SpeechGenerationSample) -> SpeechGenerationInput:
        return SpeechGenerationInput(
            text=input_sample.reference.get_transcript_string(),
            audio_name=input_sample.audio_name,
        )

    def parse_output(self, output: GeneratedAudio) -> PipelineOutput[GeneratedAudio]:
        return PipelineOutput[GeneratedAudio](prediction=output)
