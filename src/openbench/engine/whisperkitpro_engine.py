"""Unified implementation for WhisperKitPro CLI operations."""

import os
import subprocess
from pathlib import Path
from typing import Literal

import coremltools as ct
from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field


logger = get_logger(__name__)

COMPUTE_UNITS_MAPPER = {
    ct.ComputeUnit.CPU_ONLY: "cpuOnly",
    ct.ComputeUnit.CPU_AND_NE: "cpuAndNeuralEngine",
    ct.ComputeUnit.CPU_AND_GPU: "cpuAndGpu",
    ct.ComputeUnit.ALL: "all",
}


# NOTE: This is not an exhaustive list of all the possible options for
# the CLI just the ones that are most commonly used
class WhisperKitProConfig(BaseModel):
    """Configuration for transcription operations."""

    model_version: str = Field(
        "tiny",
        description="The version of the WhisperKit model to use",
    )
    model_prefix: str = Field(
        "openai",
        description="The prefix of the model to use.",
    )
    model_repo_name: str | None = Field(
        "argmaxinc/whisperkit-pro",
        description="The name of the Hugging Face model repo to use. Default is `argmaxinc/whisperkit-pro` which has Whisper checkpoints models.",
    )
    word_timestamps: bool = Field(
        True,
        description="Whether to include word timestamps in the output",
    )
    chunking_strategy: Literal["none", "vad"] = Field(
        "vad",
        description="The chunking strategy to use either `none` or `vad`",
    )
    report_path: str = Field(
        "whisperkitpro_cli_reports",
        description="The path to the directory where the report files will be saved. Defaults to `whisperkitpro_cli_reports`.",
    )
    model_vad: str | None = Field(
        None,
        description="The version of the VAD model to use",
    )
    model_vad_threshold: float | None = Field(
        None,
        description="The threshold to use for the VAD model",
    )
    audio_encoder_compute_units: ct.ComputeUnit = Field(
        ct.ComputeUnit.CPU_AND_NE,
        description="The compute units to use for the audio encoder. Default is CPU_AND_NE.",
    )
    text_decoder_compute_units: ct.ComputeUnit = Field(
        ct.ComputeUnit.CPU_AND_GPU,
        description="The compute units to use for the text decoder. Default is CPU_AND_GPU.",
    )
    diarization: bool = Field(
        False,
        description="Whether to perform diarization",
    )
    orchestration_strategy: Literal["word", "segment"] = Field(
        "segment",
        description="The orchestration strategy to use either `word` or `segment`",
    )
    speaker_models_path: str | None = Field(
        None,
        description="The path to the speaker models directory",
    )
    clusterer_version: Literal["pyannote3", "pyannote4"] = Field(
        "pyannote4",
        description="The version of the clusterer to use",
    )
    use_exclusive_reconciliation: bool = Field(
        False,
        description="Whether to use exclusive reconciliation",
    )
    fast_load: bool = Field(
        False,
        description="Whether to use fast load",
    )

    @property
    def rttm_path(self) -> str | None:
        # Path to the directory where the .rttm file with transcription should be saved
        # For some reason this is not currently being saved when --report and --diarization are provided
        return self.report_path if self.report_path is not None else "."

    def create_report_path(self) -> Path:
        report_dir = Path(self.report_path)

        report_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created report dir for WhisperKit at: {report_dir}")
        return report_dir

    def generate_cli_args(self) -> list[str]:
        args = [
            "--model",
            self.model_version,
            "--model-prefix",
            self.model_prefix,
            "--model-repo-name",
            self.model_repo_name,
            "--report",  # Always generate the report files
            "--report-path",  # Report path should always be provided
            self.report_path,
            "--chunking-strategy",
            self.chunking_strategy,
            "--audio-encoder-compute-units",
            COMPUTE_UNITS_MAPPER[self.audio_encoder_compute_units],
            "--text-decoder-compute-units",
            COMPUTE_UNITS_MAPPER[self.text_decoder_compute_units],
            "--fast-load",
            str(self.fast_load).lower(),
        ]

        # Add optional args
        if self.word_timestamps:
            args.append("--word-timestamps")
        if self.model_vad:
            args.extend(["--model-vad", self.model_vad])
        if self.model_vad_threshold:
            args.extend(["--model-vad-threshold", str(self.model_vad_threshold)])
        if self.diarization:
            args.extend(["--diarization"])
            args.extend(["--orchestration-strategy", self.orchestration_strategy])
            # Add rttm path
            args.extend(["--rttm-path", self.rttm_path])
            args.extend(["--clusterer-version", self.clusterer_version])
            # If speaker models path is provided use it
            if self.speaker_models_path:
                args.extend(["--speaker-models-path", self.speaker_models_path])
            if self.use_exclusive_reconciliation:
                args.extend(["--use-exclusive-reconciliation"])

        logger.info(f"Generating CLI args for Transcription: {args}")
        return args


class WhisperKitProInput(BaseModel):
    """Input for transcription CLI."""

    audio_path: Path
    keep_audio: bool = False
    prompt: str | None = Field(
        None,
        description="Optional prompt for keyword boosting/context"
    )
    custom_vocabulary_path: str | None = Field(
        None,
        description="Optional path to custom vocabulary file"
    )


class WhisperKitProOutput(BaseModel):
    """Output for transcription CLI."""

    json_report_path: Path = Field(
        ...,
        description="Path to the JSON report with transcription results",
    )
    srt_report_path: Path = Field(
        ...,
        description="Path to the .srt file containing transcription results",
    )
    rttm_report_path: Path | None = Field(
        ...,
        description="Path to the .rttm file containing transcription results with speaker labels assigned",
    )


class WhisperKitPro:
    """Unified CLI interface for WhisperKitPro operations."""

    def __init__(
        self,
        cli_path: str,
        transcription_config: WhisperKitProConfig,
    ) -> None:
        self.cli_path = cli_path

        self.transcription_config = transcription_config

        self.transcription_args = self.transcription_config.generate_cli_args()
        self.transcription_config.create_report_path()

    def __call__(self, input: WhisperKitProInput) -> WhisperKitProOutput:
        """Run transcription on the given audio file."""
        cmd = [
            self.cli_path,
            "transcribe",
            "--audio-path",
            str(input.audio_path),
            "--disable-keychain",  # Always disable keychain for convenience
            *self.transcription_args,
        ]

        # Add prompt for keyword boosting if provided
        if input.prompt:
            cmd.extend(["--prompt", input.prompt])

        # Add custom vocabulary path if provided
        if input.custom_vocabulary_path:
            cmd.extend(["--custom-vocabulary-path", input.custom_vocabulary_path])

        if "WHISPERKITPRO_API_KEY" in os.environ:
            cmd.extend(["--api-key", os.environ["WHISPERKITPRO_API_KEY"]])
        else:
            logger.warning(
                "`WHISPERKITPRO_API_KEY` not found in environment variables. You might run into errors if you don't have the proper permissions."
            )

        report_dir = self.transcription_config.create_report_path()
        if not report_dir:
            raise ValueError("Report directory not configured")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # Make sure to remove the api_key from the error
            error_message = e.stderr.replace(os.getenv("WHISPERKITPRO_API_KEY", ""), "********")
            raise RuntimeError(f"CLI command failed: {error_message}")

        if not input.keep_audio:
            input.audio_path.unlink(missing_ok=True)

        json_report_path = report_dir / input.audio_path.with_suffix(".json").name
        srt_report_path = report_dir / input.audio_path.with_suffix(".srt").name
        rttm_report_path = None
        if self.transcription_config.diarization:
            rttm_report_path = report_dir / input.audio_path.with_suffix(".rttm").name

        return WhisperKitProOutput(
            json_report_path=json_report_path,
            srt_report_path=srt_report_path,
            rttm_report_path=rttm_report_path,
        )
