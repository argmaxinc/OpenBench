# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import json
from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import TranscriptionSample
from ...engine.argmax_oss_engine import (
    ArgmaxOpenSourceEngine,
    ArgmaxOpenSourceEngineConfig,
    TranscriptionCliInput,
    TranscriptionCliOutput,
)
from ...pipeline_prediction import Transcript
from ..base import Pipeline, PipelineType, register_pipeline
from .common import TranscriptionConfig, TranscriptionOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("./temp_audio")
ARGMAX_OSS_DEFAULT_REPORT_PATH = "./argmax_oss_transcription_reports"


class ArgmaxOpenSourceTranscriptionConfig(TranscriptionConfig):
    """Configuration for Argmax SDK open-source CLI (`argmax-cli`) transcription."""

    cache_dir: str | None = Field(
        default=None,
        description="Cache directory for WhisperKit clone and CLI build. "
        "Defaults to ARGMAX_OSS_CACHE_DIR or ~/.cache/openbench/argmax-oss.",
    )
    commit_hash: str | None = Field(
        default=None,
        description="Optional git commit to checkout before building argmax-cli.",
    )
    cli_path: str | None = Field(
        default=None,
        description="If set, use this argmax-cli binary instead of clone/build.",
    )
    model_version: str = Field(
        default="base",
        description="Passed as --model (e.g. tiny, base, small, large-v3, large-v3-v20240930).",
    )
    model_prefix: str = Field(
        default="openai",
        description="Passed as --model-prefix.",
    )
    word_timestamps: bool = Field(
        default=True,
        description="Whether to request --word-timestamps.",
    )
    chunking_strategy: str | None = Field(
        default="vad",
        description="Chunking strategy: none or vad.",
    )
    report_path: str | None = Field(
        default=ARGMAX_OSS_DEFAULT_REPORT_PATH,
        description="Directory for JSON/SRT reports (--report-path).",
    )
    prompt: str | None = Field(
        default=None,
        description="Optional --prompt for decoding.",
    )
    text_decoder_compute_units: str = Field(
        default="cpuAndNeuralEngine",
        description="--text-decoder-compute-units",
    )
    audio_encoder_compute_units: str = Field(
        default="cpuAndNeuralEngine",
        description="--audio-encoder-compute-units",
    )

    def create_report_path(self) -> Path:
        if self.report_path is None:
            return Path.cwd()
        report_dir = Path(self.report_path)
        report_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Argmax OSS transcription report dir: %s", report_dir)
        return report_dir.resolve()

    def generate_cli_args(self) -> list[str]:
        args = [
            "--model",
            self.model_version,
            "--model-prefix",
            self.model_prefix,
            "--report",
        ]
        if self.chunking_strategy:
            args.extend(["--chunking-strategy", self.chunking_strategy])
        if self.word_timestamps:
            args.append("--word-timestamps")
        if self.report_path:
            args.extend(["--report-path", str(Path(self.report_path).resolve())])
        if self.prompt:
            args.extend(["--prompt", self.prompt])
        args.extend(
            [
                "--text-decoder-compute-units",
                self.text_decoder_compute_units,
                "--audio-encoder-compute-units",
                self.audio_encoder_compute_units,
            ]
        )
        logger.info("Argmax OSS transcribe CLI args: %s", args)
        return args


@register_pipeline
class ArgmaxOpenSourceTranscriptionPipeline(Pipeline):
    _config_class = ArgmaxOpenSourceTranscriptionConfig
    pipeline_type = PipelineType.TRANSCRIPTION

    def build_pipeline(self) -> Callable[[TranscriptionCliInput], TranscriptionCliOutput]:
        engine = ArgmaxOpenSourceEngine(
            ArgmaxOpenSourceEngineConfig(
                cache_dir=self.config.cache_dir,
                commit_hash=self.config.commit_hash,
                cli_path=self.config.cli_path,
            )
        )
        transcription_args = self.config.generate_cli_args()
        report_dir = self.config.create_report_path()

        def transcribe(inp: TranscriptionCliInput) -> TranscriptionCliOutput:
            return engine.transcribe(inp, transcription_args, report_dir)

        return transcribe

    def parse_input(self, input_sample: TranscriptionSample) -> TranscriptionCliInput:
        language = None
        if self.config.force_language:
            language = input_sample.language

        return TranscriptionCliInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            keep_audio=False,
            language=language,
        )

    def parse_output(self, output: TranscriptionCliOutput) -> TranscriptionOutput:
        with output.json_report_path.open() as f:
            data = json.load(f)

        words: list[str] = []
        start: list[float | None] = []
        end: list[float | None] = []
        for segment in data["segments"]:
            for word in segment.get("words", []):
                words.append(word["word"])
                start.append(word["start"] if "start" in word else None)
                end.append(word["end"] if "end" in word else None)

        return TranscriptionOutput(
            prediction=Transcript.from_words_info(words=words, start=start, end=end),
        )
