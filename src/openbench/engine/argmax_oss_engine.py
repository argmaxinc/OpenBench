# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Argmax SDK open-source CLI (`argmax-cli`) — clone/build, transcribe, diarize, and tts."""

import os
import subprocess
from pathlib import Path

from argmaxtools.utils import _maybe_git_clone, get_logger
from pydantic import BaseModel, Field


logger = get_logger(__name__)

ARGMAX_OSS_REPO_URL = "https://github.com/argmaxinc/argmax-oss-swift"
ARGMAX_OSS_PRODUCT = "argmax-cli"
DEFAULT_CACHE_SUBDIR = Path(".cache") / "openbench" / "argmax-oss"

# Process-wide cache of resolved CLI binary paths, keyed by (cache_root, commit_hash).
# Avoids re-cloning and re-invoking `swift build` when multiple pipelines (e.g. a
# TTS pipeline + the WER metric's transcription pipeline) build their own engine
# instances in the same run.
_CLI_PATH_CACHE: dict[tuple[str, str | None], str] = {}


def resolve_argmax_oss_cache_dir(explicit: str | Path | None = None) -> Path:
    """Absolute cache root for WhisperKit clone + `argmax-cli` build."""
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("ARGMAX_OSS_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / DEFAULT_CACHE_SUBDIR).resolve()


class ArgmaxOpenSourceEngineConfig(BaseModel):
    """Engine config: cache, optional commit pin, optional prebuilt CLI path."""

    cache_dir: str | None = Field(
        default=None,
        description="Directory for cloned repo and build artifacts (absolute after resolve). "
        "Overrides ARGMAX_OSS_CACHE_DIR when set.",
    )
    commit_hash: str | None = Field(
        default=None,
        description="Git commit to checkout in the cached WhisperKit clone.",
    )
    cli_path: str | None = Field(
        default=None,
        description="If set, skip clone/build and use this argmax-cli binary.",
    )


class TranscriptionCliInput(BaseModel):
    """Input for `argmax-cli transcribe`."""

    audio_path: Path
    keep_audio: bool = False
    language: str | None = None


class TranscriptionCliOutput(BaseModel):
    """Output paths from `argmax-cli transcribe --report`."""

    json_report_path: Path = Field(..., description="JSON report path")
    srt_report_path: Path = Field(..., description="SRT report path")
    cli_combined_output: str | None = Field(
        default=None,
        description="Concatenated stdout+stderr when transcribe was run with capture_combined_output=True.",
    )


class DiarizeCliInput(BaseModel):
    """Input for `argmax-cli diarize`."""

    audio_path: Path
    rttm_path: Path
    keep_audio: bool = False


class DiarizeCliOutput(BaseModel):
    rttm_path: Path = Field(..., description="Written RTTM path")


class TtsCliInput(BaseModel):
    """Input for `argmax-cli tts`."""

    text: str = Field(..., description="Text to synthesize.")
    output_path: Path = Field(..., description="Destination audio file path (extension picks format).")


class TtsCliOutput(BaseModel):
    """Output from `argmax-cli tts`."""

    audio_path: Path = Field(..., description="Generated audio path.")


class ArgmaxOpenSourceEngine:
    """Resolve `argmax-cli`, then run `transcribe` / `diarize` / `tts` subcommands."""

    def __init__(self, config: ArgmaxOpenSourceEngineConfig) -> None:
        self.config = config
        if config.cli_path:
            self.cli_path = str(Path(config.cli_path).expanduser().resolve())
            logger.info("Using Argmax OSS CLI at %s", self.cli_path)
            return

        cache_root = resolve_argmax_oss_cache_dir(config.cache_dir)
        cache_key = (str(cache_root), config.commit_hash)
        cached_cli_path = _CLI_PATH_CACHE.get(cache_key)
        if cached_cli_path is not None:
            logger.info("Reusing cached Argmax OSS CLI at %s", cached_cli_path)
            self.cli_path = cached_cli_path
            return

        self.cli_path = self._clone_and_build_cli(cache_root)
        _CLI_PATH_CACHE[cache_key] = self.cli_path

    def _build_cli(self, repo_dir: str) -> str:
        """Run release build (swift build -c release, not debug) and return the dir containing the binary."""
        logger.info("Building %s with: swift build -c release (not debug)", ARGMAX_OSS_PRODUCT)
        build_cmd = f"swift build -c release --product {ARGMAX_OSS_PRODUCT}"
        try:
            subprocess.run(build_cmd, cwd=repo_dir, shell=True, check=True)
            result = subprocess.run(
                f"{build_cmd} --show-bin-path",
                cwd=repo_dir,
                stdout=subprocess.PIPE,
                shell=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Build failed with return code %s", e.returncode)
            logger.error("Stdout:\n%s", getattr(e, "stdout", ""))
            logger.error("Stderr:\n%s", getattr(e, "stderr", ""))
            raise RuntimeError(
                f"Failed to build {ARGMAX_OSS_PRODUCT}: exit {e.returncode}\n"
                f"stdout: {getattr(e, 'stdout', '')}\nstderr: {getattr(e, 'stderr', '')}"
            ) from e
        bin_dir = result.stdout.strip()
        cli = Path(bin_dir) / ARGMAX_OSS_PRODUCT
        if not cli.is_file():
            raise RuntimeError(f"Expected CLI binary not found: {cli}")
        logger.info("Built Argmax OSS CLI at %s", cli)
        return bin_dir

    def _clone_and_build_cli(self, cache_root: Path) -> str:
        cache_root.mkdir(parents=True, exist_ok=True)
        repo_url_parts = ARGMAX_OSS_REPO_URL.rstrip("/").split("/")
        repo_name = repo_url_parts[-1]
        repo_owner = repo_url_parts[-2]

        logger.info("Ensuring WhisperKit clone under %s", cache_root)
        repo_dir, commit_hash = _maybe_git_clone(
            out_dir=str(cache_root),
            hub_url="github.com",
            repo_name=repo_name,
            repo_owner=repo_owner,
            commit_hash=self.config.commit_hash,
        )
        self.config.commit_hash = commit_hash
        logger.info("%s at commit %s — running sanity build", repo_name, commit_hash)
        bin_dir = self._build_cli(repo_dir)
        return str(Path(bin_dir) / ARGMAX_OSS_PRODUCT)

    def transcribe(
        self,
        input: TranscriptionCliInput,
        transcription_args: list[str],
        report_dir: Path,
        capture_combined_output: bool = False,
    ) -> TranscriptionCliOutput:
        """Run `argmax-cli transcribe` with pre-built flag list (see transcription config)."""
        cmd = [
            self.cli_path,
            "transcribe",
            "--audio-path",
            str(input.audio_path),
            *transcription_args,
        ]
        if input.language:
            cmd.extend(["--language", input.language])

        logger.debug("Argmax OSS transcribe: %s", cmd)
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"argmax-cli transcribe failed: {e.stderr}") from e

        json_report_path = report_dir / f"{input.audio_path.stem}.json"
        srt_report_path = report_dir / f"{input.audio_path.stem}.srt"

        cli_combined_output: str | None = None
        if capture_combined_output:
            cli_combined_output = (result.stdout or "") + (result.stderr or "")

        if not input.keep_audio:
            input.audio_path.unlink(missing_ok=True)

        return TranscriptionCliOutput(
            json_report_path=json_report_path,
            srt_report_path=srt_report_path,
            cli_combined_output=cli_combined_output,
        )

    def diarize(self, input: DiarizeCliInput, diarize_args: list[str]) -> DiarizeCliOutput:
        """Run `argmax-cli diarize` with pre-built flag list (see diarization config)."""
        input.rttm_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.cli_path,
            "diarize",
            "--audio-path",
            str(input.audio_path),
            "--rttm-path",
            str(input.rttm_path),
            *diarize_args,
        ]
        logger.debug("Argmax OSS diarize: %s", cmd)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"argmax-cli diarize failed: {e.stderr}") from e

        if not input.keep_audio:
            input.audio_path.unlink(missing_ok=True)

        return DiarizeCliOutput(rttm_path=input.rttm_path)

    def tts(self, input: TtsCliInput, tts_args: list[str]) -> TtsCliOutput:
        """Run `argmax-cli tts` with pre-built flag list (see TTS pipeline config)."""
        input.output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.cli_path,
            "tts",
            "--text",
            input.text,
            "--output-path",
            str(input.output_path),
            *tts_args,
        ]
        logger.debug("Argmax OSS tts: %s", cmd)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"argmax-cli tts failed: {e.stderr}") from e

        if not input.output_path.exists():
            raise RuntimeError(f"argmax-cli tts returned 0 but no audio at {input.output_path}")

        return TtsCliOutput(audio_path=input.output_path)
