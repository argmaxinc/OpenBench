# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Orchestration: `argmax-cli diarize` then `transcribe`, merge speakers by word timestamps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset import OrchestrationSample
from ...engine.argmax_oss_engine import (
    ArgmaxOpenSourceEngine,
    ArgmaxOpenSourceEngineConfig,
    DiarizeCliInput,
    TranscriptionCliInput,
)
from ...pipeline_prediction import DiarizationAnnotation, Transcript, Word
from ..base import Pipeline, PipelineType, register_pipeline
from ..diarization.common import DiarizationOutput
from ..transcription.common import TranscriptionOutput
from .common import OrchestrationConfig, OrchestrationOutput


logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("./temp_audio_argmax_oss_orch")
ARGMAX_OSS_ORCH_DEFAULT_REPORT = "./argmax_oss_orchestration_reports"


def _merge_transcript_with_diarization(transcript: Transcript, diar: DiarizationAnnotation) -> Transcript:
    """Assign each word a speaker label from RTTM segments using word midpoint time."""
    new_words: list[Word] = []
    for w in transcript.words:
        t: float | None = None
        if w.start is not None and w.end is not None:
            t = (w.start + w.end) / 2.0
        elif w.start is not None:
            t = w.start
        elif w.end is not None:
            t = w.end
        speaker: str | None = None
        if t is not None:
            for segment, _, label in diar.itertracks(yield_label=True):
                if segment.start <= t <= segment.end:
                    speaker = label
                    break
        new_words.append(Word(word=w.word, start=w.start, end=w.end, speaker=speaker))
    return Transcript(words=new_words)


class ArgmaxOpenSourceOrchestrationConfig(OrchestrationConfig):
    cache_dir: str | None = Field(
        default=None,
        description="Cache directory for WhisperKit clone / argmax-cli build.",
    )
    commit_hash: str | None = Field(default=None, description="Optional git commit pin.")
    cli_path: str | None = Field(default=None, description="Prebuilt argmax-cli binary.")
    model_version: str = Field(default="base", description="--model for transcribe.")
    model_prefix: str = Field(default="openai", description="--model-prefix.")
    word_timestamps: bool = Field(default=True, description="Must be true for speaker merge.")
    chunking_strategy: str | None = Field(default="vad")
    report_path: str | None = Field(
        default=ARGMAX_OSS_ORCH_DEFAULT_REPORT,
        description="Report directory for JSON/SRT and RTTM.",
    )
    prompt: str | None = None
    text_decoder_compute_units: str = Field(default="cpuAndNeuralEngine")
    audio_encoder_compute_units: str = Field(default="cpuAndNeuralEngine")
    # diarize
    diarization_model_path: str | None = Field(default=None, description="--model-path for diarize.")
    diarization_model_repo: str | None = Field(default=None)
    diarization_model_token: str | None = Field(default=None)
    diarization_download_model_path: str | None = Field(default=None)
    cluster_distance_threshold: float | None = Field(default=None)
    use_exclusive_reconciliation: bool = Field(default=False)
    disable_full_redundancy: bool = Field(default=False)
    diarization_verbose: bool = Field(default=False)
    include_sub_outputs: bool = Field(
        default=False,
        description="If true, populate diarization_output and transcription_output on OrchestrationOutput.",
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

    def generate_diarize_cli_args(self) -> list[str]:
        args: list[str] = []
        if self.diarization_model_path:
            args.extend(["--model-path", self.diarization_model_path])
        if self.diarization_model_repo:
            args.extend(["--model-repo", self.diarization_model_repo])
        if self.diarization_model_token:
            args.extend(["--model-token", self.diarization_model_token])
        if self.diarization_download_model_path:
            args.extend(["--download-model-path", self.diarization_download_model_path])
        if self.cluster_distance_threshold is not None:
            args.extend(["--cluster-distance-threshold", str(self.cluster_distance_threshold)])
        if self.use_exclusive_reconciliation:
            args.append("--use-exclusive-reconciliation")
        if self.disable_full_redundancy:
            args.append("--disable-full-redundancy")
        if self.diarization_verbose:
            args.append("--verbose")
        return args


class ArgmaxOpenSourceOrchestrationPipelineInput(BaseModel):
    audio_path: Path
    language: str | None = None


@register_pipeline
class ArgmaxOpenSourceOrchestrationPipeline(Pipeline):
    _config_class = ArgmaxOpenSourceOrchestrationConfig
    pipeline_type = PipelineType.ORCHESTRATION

    def build_pipeline(self) -> Callable[[ArgmaxOpenSourceOrchestrationPipelineInput], OrchestrationOutput]:
        engine = ArgmaxOpenSourceEngine(
            ArgmaxOpenSourceEngineConfig(
                cache_dir=self.config.cache_dir,
                commit_hash=self.config.commit_hash,
                cli_path=self.config.cli_path,
            )
        )
        report_dir = self.config.create_report_path()
        transcribe_args = self.config.generate_transcribe_cli_args(report_dir)
        diarize_args_base = self.config.generate_diarize_cli_args()
        include_sub = self.config.include_sub_outputs

        def run(inp: ArgmaxOpenSourceOrchestrationPipelineInput) -> OrchestrationOutput:
            rttm_path = report_dir / f"{inp.audio_path.stem}.rttm"
            engine.diarize(
                DiarizeCliInput(audio_path=inp.audio_path, rttm_path=rttm_path, keep_audio=True),
                list(diarize_args_base),
            )
            t_out = engine.transcribe(
                TranscriptionCliInput(audio_path=inp.audio_path, keep_audio=False, language=inp.language),
                transcribe_args,
                report_dir,
            )
            with t_out.json_report_path.open() as f:
                data = json.load(f)
            words: list[str] = []
            start: list[float | None] = []
            end: list[float | None] = []
            for segment in data["segments"]:
                for word in segment.get("words", []):
                    words.append(word["word"])
                    start.append(word["start"] if "start" in word else None)
                    end.append(word["end"] if "end" in word else None)
            transcript = Transcript.from_words_info(words=words, start=start, end=end)
            diar = DiarizationAnnotation.load_annotation_file(str(rttm_path))
            merged = _merge_transcript_with_diarization(transcript, diar)

            diar_out: DiarizationOutput | None = None
            tr_out: TranscriptionOutput | None = None
            if include_sub:
                diar_out = DiarizationOutput(prediction=diar)
                tr_out = TranscriptionOutput(prediction=transcript)

            return OrchestrationOutput(
                prediction=merged,
                diarization_output=diar_out,
                transcription_output=tr_out,
            )

        return run

    def parse_input(self, input_sample: OrchestrationSample) -> ArgmaxOpenSourceOrchestrationPipelineInput:
        language = None
        if self.config.force_language:
            language = input_sample.language
        return ArgmaxOpenSourceOrchestrationPipelineInput(
            audio_path=input_sample.save_audio(TEMP_AUDIO_DIR),
            language=language,
        )

    def parse_output(self, output: OrchestrationOutput) -> OrchestrationOutput:
        return output
