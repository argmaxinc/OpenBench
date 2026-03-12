# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""SpeakerKit OSS diarization pipeline using whisperkit-cli diarize command.

Uses the open-source WhisperKit CLI (no API key). All CLI arguments are exposed
in config; only defaults are used in the alias.
"""

import re
import subprocess
from pathlib import Path
from typing import Callable, TypedDict

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import DiarizationSample
from ...pipeline_prediction import DiarizationAnnotation
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig


__all__ = ["SpeakerKitOssPipeline", "SpeakerKitOssPipelineConfig"]

logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("audio_temp")


class SpeakerKitOssPipelineConfig(DiarizationPipelineConfig):
    """Configuration for SpeakerKit OSS (whisperkit-cli diarize).

    All whisperkit-cli diarize options are exposed. Defaults match CLI defaults.
    """

    cli_path: str = Field(..., description="The absolute path to the whisperkit-cli binary")
    model_path: str | None = Field(
        default=None,
        description="Path of local model files (skips download)",
    )
    model_repo: str | None = Field(
        default=None,
        description="HuggingFace model repository",
    )
    model_token: str | None = Field(
        default=None,
        description="HuggingFace API token",
    )
    download_model_path: str | None = Field(
        default=None,
        description="Path to save downloaded models",
    )
    num_speakers: int | None = Field(
        default=None,
        description="Number of speakers to detect (default: automatic)",
    )
    cluster_distance_threshold: float | None = Field(
        default=None,
        description="Cluster distance threshold for VBx clustering (default: 0.6)",
    )
    use_exclusive_reconciliation: bool = Field(
        default=False,
        description="Use exclusive reconciliation in post processing",
    )
    disable_full_redundancy: bool = Field(
        default=False,
        description="Disable full redundancy in segmenter",
    )
    verbose: bool = Field(default=True, description="Enable verbose output")


class SpeakerKitOssInput(TypedDict):
    audio_path: Path
    output_path: Path
    num_speakers_override: int | None


class SpeakerKitOssCli:
    def __init__(self, config: SpeakerKitOssPipelineConfig):
        self.config = config

    def __call__(self, pipeline_input: SpeakerKitOssInput) -> tuple[Path, float]:
        cmd = [
            self.config.cli_path,
            "diarize",
            "--audio-path",
            str(pipeline_input["audio_path"]),
            "--rttm-path",
            str(pipeline_input["output_path"]),
        ]

        if self.config.model_path is not None:
            cmd.extend(["--model-path", self.config.model_path])
        if self.config.model_repo is not None:
            cmd.extend(["--model-repo", self.config.model_repo])
        if self.config.model_token is not None:
            cmd.extend(["--model-token", self.config.model_token])
        if self.config.download_model_path is not None:
            cmd.extend(["--download-model-path", self.config.download_model_path])

        num_speakers = pipeline_input["num_speakers_override"]
        if num_speakers is None:
            num_speakers = self.config.num_speakers
        if num_speakers is not None:
            cmd.extend(["--num-speakers", str(num_speakers)])

        if self.config.cluster_distance_threshold is not None:
            cmd.extend(
                [
                    "--cluster-distance-threshold",
                    str(self.config.cluster_distance_threshold),
                ]
            )
        if self.config.use_exclusive_reconciliation:
            cmd.append("--use-exclusive-reconciliation")
        if self.config.disable_full_redundancy:
            cmd.append("--disable-full-redundancy")
        if self.config.verbose:
            cmd.append("--verbose")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"Diarization CLI stdout:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Diarization CLI failed with error: {e.stderr}") from e

        pipeline_input["audio_path"].unlink()

        # Parse "Total Time: ... ms" from stdout (tabs or spaces)
        pattern = r"Total Time:\s+(\d+\.\d+)\s+ms"
        matches = re.search(pattern, result.stdout)
        total_time_ms = float(matches.group(1)) if matches else 0.0

        return pipeline_input["output_path"], total_time_ms / 1000


@register_pipeline
class SpeakerKitOssPipeline(Pipeline):
    _config_class = SpeakerKitOssPipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(self) -> Callable[[SpeakerKitOssInput], tuple[Path, float]]:
        return SpeakerKitOssCli(self.config)

    def parse_input(self, input_sample: DiarizationSample) -> SpeakerKitOssInput:
        inputs: SpeakerKitOssInput = {
            "audio_path": input_sample.save_audio(TEMP_AUDIO_DIR),
            "output_path": input_sample.audio_name + ".rttm",
            "num_speakers_override": None,
        }
        if self.config.use_exact_num_speakers:
            inputs["num_speakers_override"] = len(set(input_sample.annotation.speakers))
        return inputs

    def parse_output(self, output: tuple[Path, float]) -> DiarizationOutput:
        prediction = DiarizationAnnotation.load_annotation_file(output[0])
        return DiarizationOutput(prediction=prediction, prediction_time=output[1])
