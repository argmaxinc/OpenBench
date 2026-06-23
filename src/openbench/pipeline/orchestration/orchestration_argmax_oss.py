# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Orchestration: `argmax-cli transcribe --diarization`; prediction from RTTM-like verbose log (word + speaker, no timestamps)."""

from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset import OrchestrationSample
from ...engine.argmax_oss_engine import (
    ArgmaxOpenSourceEngine,
    ArgmaxOpenSourceEngineConfig,
    TranscriptionCliInput,
)
from ...pipeline_prediction import Transcript, Word
from ..base import Pipeline, PipelineType, register_pipeline
from .common import OrchestrationConfig, OrchestrationOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("./temp_audio_argmax_oss_orch")
ARGMAX_OSS_ORCH_DEFAULT_REPORT = "./argmax_oss_orchestration_reports"

SPEAKER_DIARIZATION_MARKER = "---- Speaker Diarization Results ----"
_MIN_SPEAKER_LINE_PARTS = 9


def _slice_rttm_like_block(cli_log: str) -> str:
    if SPEAKER_DIARIZATION_MARKER not in cli_log:
        logger.warning(
            "Diarization marker %r not found in CLI output; using empty RTTM-like block",
            SPEAKER_DIARIZATION_MARKER,
        )
        return ""
    return cli_log.split(SPEAKER_DIARIZATION_MARKER, 1)[1].strip()


def _words_from_rttm_like_text(text: str) -> list[Word]:
    words: list[Word] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < _MIN_SPEAKER_LINE_PARTS or parts[0] != "SPEAKER":
            continue
        speaker = parts[-3]
        transcript_words = parts[5:-4]
        words.extend(Word(word=w, speaker=speaker, start=None, end=None) for w in transcript_words)
    return words


class ArgmaxOpenSourceOrchestrationConfig(OrchestrationConfig):
    cache_dir: str | None = Field(
        default=None,
        description="Cache directory for WhisperKit clone / argmax-cli build.",
    )
    commit_hash: str | None = Field(default=None, description="Optional git commit pin.")
    cli_path: str | None = Field(default=None, description="Prebuilt argmax-cli binary.")
    model_version: str = Field(default="base", description="--model for transcribe.")
    model_prefix: str = Field(default="openai", description="--model-prefix.")
    word_timestamps: bool = Field(
        default=False,
        description="--word-timestamps on transcribe (optional; affects JSON/SRT written under report_path only).",
    )
    chunking_strategy: str | None = Field(default="vad")
    report_path: str | None = Field(
        default=ARGMAX_OSS_ORCH_DEFAULT_REPORT,
        description="Report directory for JSON/SRT (--report-path).",
    )
    prompt: str | None = None
    text_decoder_compute_units: str = Field(default="cpuAndNeuralEngine")
    audio_encoder_compute_units: str = Field(default="cpuAndNeuralEngine")
    diarization_model_path: str | None = Field(
        default=None,
        description="Optional --diarization-model-path for transcribe.",
    )
    diarization_model_repo: str | None = Field(
        default=None,
        description="Optional --diarization-model-repo for transcribe.",
    )

    def create_report_path(self) -> Path:
        path = Path(self.report_path or ".").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def generate_transcribe_cli_args(self, report_dir: Path) -> list[str]:
        args = [
            "--model",
            self.model_version,
            "--model-prefix",
            self.model_prefix,
            "--report",
            "--chunking-strategy",
            str(self.chunking_strategy or "vad"),
        ]
        if self.word_timestamps:
            args.append("--word-timestamps")
        args.extend(
            [
                "--report-path",
                str(report_dir),
                "--text-decoder-compute-units",
                self.text_decoder_compute_units,
                "--audio-encoder-compute-units",
                self.audio_encoder_compute_units,
            ]
        )
        if self.prompt:
            args.extend(["--prompt", self.prompt])
        return args

    def generate_transcribe_diarization_args(self) -> list[str]:
        args: list[str] = ["--diarization", "--verbose"]
        if self.diarization_model_path:
            args.extend(["--diarization-model-path", self.diarization_model_path])
        if self.diarization_model_repo:
            args.extend(["--diarization-model-repo", self.diarization_model_repo])
        return args


class ArgmaxOpenSourceOrchestrationPipelineInput(BaseModel):
    audio_path: Path
    language: str | None = None


@register_pipeline
class ArgmaxOpenSourceOrchestrationPipeline(Pipeline):
    _config_class = ArgmaxOpenSourceOrchestrationConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(
        self,
    ) -> Callable[[ArgmaxOpenSourceOrchestrationPipelineInput], tuple[str, Path]]:
        engine = ArgmaxOpenSourceEngine(
            ArgmaxOpenSourceEngineConfig(
                cache_dir=self.config.cache_dir,
                commit_hash=self.config.commit_hash,
                cli_path=self.config.cli_path,
            )
        )
        report_dir = self.config.create_report_path()
        transcribe_args = (
            self.config.generate_transcribe_cli_args(report_dir) + self.config.generate_transcribe_diarization_args()
        )

        def run(inp: ArgmaxOpenSourceOrchestrationPipelineInput) -> tuple[str, Path]:
            t_out = engine.transcribe(
                TranscriptionCliInput(audio_path=inp.audio_path, keep_audio=False, language=inp.language),
                transcribe_args,
                report_dir,
                capture_combined_output=True,
            )
            log = t_out.cli_combined_output or ""
            rttm_like = _slice_rttm_like_block(log)
            return (rttm_like, t_out.json_report_path)

        return run

    def parse_input(self, input_sample: OrchestrationSample) -> ArgmaxOpenSourceOrchestrationPipelineInput:
        language = None
        if self.config.force_language:
            language = input_sample.language
        return ArgmaxOpenSourceOrchestrationPipelineInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            language=language,
        )

    def parse_output(self, output: tuple[str, Path]) -> OrchestrationOutput:
        rttm_like_text, _ = output
        rttm_words = _words_from_rttm_like_text(rttm_like_text)
        return OrchestrationOutput(
            prediction=Transcript(words=rttm_words),
            diarization_output=None,
            transcription_output=None,
        )
