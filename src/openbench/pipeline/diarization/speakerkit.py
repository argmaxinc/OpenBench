# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Literal, TypedDict

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset import DiarizationSample
from ...pipeline_prediction import DiarizationAnnotation
from ..base import Pipeline, PipelineType, register_pipeline
from .common import DiarizationOutput, DiarizationPipelineConfig


__all__ = ["SpeakerKitPipeline", "SpeakerKitPipelineConfig"]

logger = get_logger(__name__)

TEMP_AUDIO_DIR = Path("audio_temp")


class SpeakerKitPipelineConfig(DiarizationPipelineConfig):
    cli_path: str = Field(..., description="The absolute path to the SpeakerKit CLI")
    clusterer_version: Literal["pyannote3", "pyannote4"] = Field(
        "pyannote4", description="The version of the clusterer to use"
    )
    model_path: str | None = Field(None, description="The absolute path to the SpeakerKit model")


class SpeakerKitInput(TypedDict):
    audio_path: Path
    output_path: Path
    num_speakers: int | None


class SpeakerKitCli:
    def __init__(self, config: SpeakerKitPipelineConfig):
        self.cli_path = config.cli_path
        self.model_path = config.model_path
        self.clusterer_version = config.clusterer_version

    def __call__(self, speakerkit_input: SpeakerKitInput) -> tuple[Path, float]:
        cmd = [
            self.cli_path,
            "diarize",
            "--audio-path",
            str(speakerkit_input["audio_path"]),
            "--rttm-path",
            str(speakerkit_input["output_path"]),
            "--clusterer-version",
            self.clusterer_version,
            "--verbose",
        ]

        if self.model_path:
            cmd.extend(["--model-path", self.model_path])

        if speakerkit_input["num_speakers"] is not None:
            cmd.extend(["--num-speakers", str(speakerkit_input["num_speakers"])])

        if "SPEAKERKIT_API_KEY" in os.environ:
            cmd.extend(["--api-key", os.environ["SPEAKERKIT_API_KEY"]])

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Diarization CLI stdout:\n{result.stdout}")
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
        pattern = r"Model Load Time:\s+\d+\.\d+\s+ms\nTotal Time:\s+(\d+\.\d+)\s+ms"
        matches = re.search(pattern, result.stdout)
        total_time = float(matches.group(1))

        return speakerkit_input["output_path"], total_time / 1000


@register_pipeline
class SpeakerKitPipeline(Pipeline):
    _config_class = SpeakerKitPipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def build_pipeline(self) -> Callable[[SpeakerKitInput], tuple[Path, float]]:
        return SpeakerKitCli(cli_path=self.config.cli_path, model_path=self.config.model_path)

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
