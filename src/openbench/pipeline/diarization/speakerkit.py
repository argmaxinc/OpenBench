# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Literal, TypedDict

from argmaxtools.utils import get_logger
from pydantic import Field, model_validator

from ...dataset import DiarizationSample
from ...pipeline_prediction import DiarizationAnnotation
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig


__all__ = ["SpeakerKitPipeline", "SpeakerKitPipelineConfig"]

logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("audio_temp")


class SpeakerKitInput(TypedDict):
    audio_path: Path
    output_path: Path
    num_speakers: int | None


class SpeakerKitPipelineConfig(DiarizationPipelineConfig):
    cli_path: str = Field(..., description="The absolute path to the SpeakerKit CLI")
    model_path: str | None = Field(None, description="The absolute path to the SpeakerKit model directory")
    clusterer_version: Literal["pyannote3", "pyannote4", "sortformer"] = Field(
        "pyannote4", description="The version of the clusterer to use"
    )
    sortformer_model_name: str | None = Field(None, description="The name of the Sortformer model to use")
    sortformer_model_variant: str | None = Field(None, description="The variant of the Sortformer model to use")

    @model_validator(mode="after")
    def validate_sortformer_model(self) -> "SpeakerKitPipelineConfig":
        if self.sortformer_model_name is not None and self.sortformer_model_variant is None:
            raise ValueError(
                "If `sortformer_model_name` is provided, `sortformer_model_variant` must also be provided"
            )

        if self.sortformer_model_name is None and self.sortformer_model_variant is not None:
            raise ValueError(
                "If `sortformer_model_variant` is provided, `sortformer_model_name` must also be provided"
            )

        return self

    @property
    def is_sortformer(self) -> bool:
        return self.clusterer_version == "sortformer"

    def generate_cli_args(self, inputs: SpeakerKitInput) -> list[str]:
        cmd = [
            self.cli_path,
            "diarize",
            "--audio-path",
            str(inputs["audio_path"]),
            "--rttm-path",
            str(inputs["output_path"]),
            "--clusterer-version",
            self.clusterer_version,
            "--verbose",
        ]

        # Only check variant as we already checked both should be provided
        if self.sortformer_model_variant is not None:
            cmd.extend(
                [
                    "--sortformer-model-name",
                    self.sortformer_model_name,
                    "--sortformer-model-variant",
                    self.sortformer_model_variant,
                ]
            )

        if self.model_path is not None:
            cmd.extend(["--model-path", self.model_path])

        if inputs["num_speakers"] is not None:
            if self.is_sortformer:
                logger.warning("`num_speakers` is not supported for Sortformer. Ignoring...")
            else:
                cmd.extend(["--num-speakers", str(inputs["num_speakers"])])

        if "SPEAKERKIT_API_KEY" in os.environ:
            cmd.extend(["--api-key", os.environ["SPEAKERKIT_API_KEY"]])

        return cmd

    def parse_stdout(self, stdout: str) -> float:
        # Default pattern for pyannote models
        pattern = r"Model Load Time:\s+\d+\.\d+\s+ms\nTotal Time:\s+(\d+\.\d+)\s+ms"
        divisor = 1000.0

        # if model is sortfomer we override the pattern and divisor
        if self.is_sortformer:
            pattern = r"Prediction time:\s+(\d+\.\d+)\s+seconds"
            divisor = 1.0

        matches = re.search(pattern, stdout)
        if matches is None:
            raise ValueError(f"Could not parse prediction time from stdout: {stdout!r}")
        return float(matches.group(1)) / divisor


class SpeakerKitCli:
    def __init__(self, config: SpeakerKitPipelineConfig):
        self.config = config

    def __call__(self, speakerkit_input: SpeakerKitInput) -> tuple[Path, float]:
        cmd = self.config.generate_cli_args(speakerkit_input)

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"Diarization CLI stdout:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            # Strip api-key from stderr if ``SPEAKERKIT_API_KEY`` is set
            if "SPEAKERKIT_API_KEY" in os.environ:
                stderr = e.stderr.replace(os.environ["SPEAKERKIT_API_KEY"], "***")
            else:
                stderr = e.stderr

            raise RuntimeError(f"Diarization CLI failed with error: {stderr}") from e

        # Delete the audio file
        speakerkit_input["audio_path"].unlink()

        # Parse stdout and take the total time it took to diarize
        total_time = self.config.parse_stdout(result.stdout)

        return speakerkit_input["output_path"], total_time


@register_pipeline
class SpeakerKitPipeline(Pipeline):
    _config_class = SpeakerKitPipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(self) -> Callable[[SpeakerKitInput], tuple[Path, float]]:
        return SpeakerKitCli(self.config)

    def parse_input(self, input_sample: DiarizationSample) -> SpeakerKitInput:
        inputs: SpeakerKitInput = {
            "audio_path": input_sample.save_audio(TEMP_AUDIO_DIR),
            "output_path": input_sample.audio_name + ".rttm",
            "num_speakers": None,
        }
        if self.config.use_exact_num_speakers:
            inputs["num_speakers"] = len(set(input_sample.annotation.speakers))

        return inputs

    def parse_output(self, output: tuple[Path, float]) -> DiarizationOutput:
        prediction = DiarizationAnnotation.load_annotation_file(output[0])
        return DiarizationOutput(prediction=prediction, prediction_time=output[1])
