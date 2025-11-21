"""Unified implementation for WhisperKitPro CLI operations."""

import os
import subprocess
from pathlib import Path
from typing import Literal

import coremltools as ct
from argmaxtools.utils import get_logger
from huggingface_hub import snapshot_download
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
    """Configuration for transcription operations.

    Supports two modes:
    1. Legacy: model_version, model_prefix, model_repo_name
    2. New: repo_id, model_variant (downloads models locally)
    """

    # Legacy fields
    model_version: str | None = Field(
        None,
        description="(Legacy) WhisperKit model version",
    )
    model_prefix: str | None = Field(
        None,
        description="(Legacy) Model prefix",
    )
    model_repo_name: str | None = Field(
        None,
        description="(Legacy) HuggingFace model repo name",
    )

    # New fields for model download
    repo_id: str | None = Field(
        None,
        description="HuggingFace repo ID",
    )
    model_variant: str | None = Field(
        None,
        description="Model variant folder name",
    )
    models_cache_dir: str | None = Field(
        None,
        description="Directory to cache downloaded models",
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

    def generate_cli_args(self, model_path: Path | None = None) -> list[str]:
        # Use either --model-path (new) or legacy model args
        if self.use_model_path:
            if model_path is None:
                raise ValueError("model_path required when using repo_id/model_variant")
            args = [
                "--model-path",
                str(model_path),
            ]
        else:
            # Legacy mode
            args = [
                "--model",
                self.model_version,
                "--model-prefix",
                self.model_prefix,
                "--model-repo-name",
                self.model_repo_name,
            ]

        # Common args
        args.extend(
            [
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
        )

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

    @property
    def use_model_path(self) -> bool:
        """Check if we should use --model-path vs legacy args."""
        return self.repo_id is not None and self.model_variant is not None

    def download_and_prepare_model(self) -> Path:
        """Download model from HuggingFace and prepare folder.

        Returns:
            Path to model directory for --model-path
        """
        if not self.use_model_path:
            raise ValueError("download_and_prepare_model requires repo_id and model_variant")

        cache_dir = Path(self.models_cache_dir or "./models_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Model path: cache_dir / repo_id / model_variant
        repo_dir = self.repo_id.replace("/", "_")
        model_path = cache_dir / repo_dir / self.model_variant

        # Check if model already exists
        if model_path.exists():
            logger.info(f"Model already exists at: {model_path}")
            return model_path

        logger.info(f"Downloading model from {self.repo_id}, variant: {self.model_variant}")

        # Download specific model variant folder from HuggingFace
        try:
            downloaded_path = snapshot_download(
                repo_id=self.repo_id,
                allow_patterns=f"{self.model_variant}/*",
                local_dir=cache_dir / repo_dir,
                local_dir_use_symlinks=False,
            )

            logger.info(f"Model downloaded to: {downloaded_path}")
            logger.info(f"Model path for CLI: {model_path}")

            if not model_path.exists():
                raise RuntimeError(f"Model download succeeded but path doesn't exist: {model_path}")

            return model_path

        except Exception as e:
            raise RuntimeError(f"Failed to download model from {self.repo_id}: {e}") from e


class WhisperKitProInput(BaseModel):
    """Input for transcription CLI."""

    audio_path: Path
    keep_audio: bool = False
    custom_vocabulary_path: str | None = Field(None, description="Optional path to custom vocabulary file")


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

        # Download and prepare model if using new model management
        self.model_path = None
        if self.transcription_config.use_model_path:
            logger.info("Using new model management with repo_id/model_variant")
            self.model_path = self.transcription_config.download_and_prepare_model()
        else:
            logger.info("Using legacy model management")

        # Generate CLI args (with model_path if available)
        self.transcription_args = self.transcription_config.generate_cli_args(model_path=self.model_path)
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
