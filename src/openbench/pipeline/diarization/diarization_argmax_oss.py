# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Diarization via Argmax SDK open-source `argmax-cli diarize`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset import DiarizationSample
from ...engine.argmax_oss_engine import (
    ArgmaxOpenSourceEngine,
    ArgmaxOpenSourceEngineConfig,
    DiarizeCliInput,
    DiarizeCliOutput,
)
from ...pipeline_prediction import DiarizationAnnotation
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig


logger = get_logger(__name__)

__all__ = [
    "ArgmaxOpenSourceDiarizationConfig",
    "ArgmaxOpenSourceDiarizationPipeline",
    "ArgmaxOpenSourceDiarizePipelineInput",
]

TEMP_AUDIO_DIR = Path("audio_temp_argmax_oss")


class ArgmaxOpenSourceDiarizationConfig(DiarizationPipelineConfig):
    cache_dir: str | None = Field(
        default=None,
        description="Cache directory for WhisperKit clone and CLI build. "
        "Defaults to ARGMAX_OSS_CACHE_DIR or ~/.cache/openbench/argmax-oss.",
    )
    commit_hash: str | None = Field(default=None, description="Optional git commit pin for the clone.")
    cli_path: str | None = Field(default=None, description="Prebuilt argmax-cli path; skips clone/build.")
    model_path: str | None = Field(default=None, description="--model-path (local diarization models).")
    model_repo: str | None = Field(default=None, description="--model-repo (HuggingFace).")
    model_token: str | None = Field(default=None, description="--model-token (optional; scrubbed from errors).")
    download_model_path: str | None = Field(default=None, description="--download-model-path.")
    cluster_distance_threshold: float | None = Field(
        default=None,
        description="--cluster-distance-threshold (default on CLI is 0.6).",
    )
    use_exclusive_reconciliation: bool = Field(default=False, description="--use-exclusive-reconciliation.")
    disable_full_redundancy: bool = Field(default=False, description="--disable-full-redundancy.")
    verbose: bool = Field(default=False, description="--verbose.")
    num_speakers: int | None = Field(
        default=None,
        description="Optional static --num-speakers. Ignored when use_exact_num_speakers derives count from reference.",
    )

    def generate_diarize_cli_args(self) -> list[str]:
        args: list[str] = []
        if self.model_path:
            args.extend(["--model-path", self.model_path])
        if self.model_repo:
            args.extend(["--model-repo", self.model_repo])
        if self.model_token:
            args.extend(["--model-token", self.model_token])
        if self.download_model_path:
            args.extend(["--download-model-path", self.download_model_path])
        if self.cluster_distance_threshold is not None:
            args.extend(["--cluster-distance-threshold", str(self.cluster_distance_threshold)])
        if self.use_exclusive_reconciliation:
            args.append("--use-exclusive-reconciliation")
        if self.disable_full_redundancy:
            args.append("--disable-full-redundancy")
        if self.verbose:
            args.append("--verbose")
        return args


class ArgmaxOpenSourceDiarizePipelineInput(BaseModel):
    audio_path: Path
    rttm_path: Path
    num_speakers: int | None = None


@register_pipeline
class ArgmaxOpenSourceDiarizationPipeline(Pipeline):
    _config_class = ArgmaxOpenSourceDiarizationConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(self) -> Callable[[ArgmaxOpenSourceDiarizePipelineInput], DiarizeCliOutput]:
        engine = ArgmaxOpenSourceEngine(
            ArgmaxOpenSourceEngineConfig(
                cache_dir=self.config.cache_dir,
                commit_hash=self.config.commit_hash,
                cli_path=self.config.cli_path,
            )
        )
        diarize_args_base = self.config.generate_diarize_cli_args()

        def run(inp: ArgmaxOpenSourceDiarizePipelineInput) -> DiarizeCliOutput:
            args = list(diarize_args_base)
            if inp.num_speakers is not None:
                args.extend(["--num-speakers", str(inp.num_speakers)])
            return engine.diarize(
                DiarizeCliInput(
                    audio_path=inp.audio_path,
                    rttm_path=inp.rttm_path,
                    keep_audio=False,
                ),
                args,
            )

        return run

    def parse_input(self, input_sample: DiarizationSample) -> ArgmaxOpenSourceDiarizePipelineInput:
        audio_path = input_sample.save_audio(TEMP_AUDIO_DIR)
        rttm_path = audio_path.with_suffix(".rttm")
        num_speakers: int | None = self.config.num_speakers
        if self.config.use_exact_num_speakers:
            num_speakers = len(set(input_sample.annotation.speakers))
        return ArgmaxOpenSourceDiarizePipelineInput(
            audio_path=audio_path,
            rttm_path=rttm_path,
            num_speakers=num_speakers,
        )

    def parse_output(self, output: DiarizeCliOutput) -> DiarizationOutput:
        prediction = DiarizationAnnotation.load_annotation_file(str(output.rttm_path))
        return DiarizationOutput(prediction=prediction)
